"""
Main uncertainty estimator for LLM answer correctness prediction.

Combines token-level statistics and layer activation patterns to estimate
whether an LLM's answer is likely correct or incorrect.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from feature_extractor import FeatureExtractor
from utils import get_default_layers, format_prompt


class UncertaintyEstimator:
    """
    Estimate answer correctness using token and activation features.

    Note: When trained on small models with low base accuracy (<20%),
    predictions are biased toward 'incorrect' due to class imbalance.
    """

    # Feature weights for token-based uncertainty
    TOKEN_WEIGHTS = {
        'mean_margin': 0.22,
        'min_margin': 0.18,
        'pct_rank_gt_2': 0.12,
        'q1_confidence': 0.10,
        'content_mean_confidence': 0.10,
        'confidence_q4_minus_q1': 0.06,
        'confidence_trend': 0.06,
        'mean_confidence': 0.08,
        'std_confidence': 0.04,
        'last_token_confidence': 0.04,
    }

    # Feature weights for activation-based uncertainty
    ACTIVATION_WEIGHTS = {
        'norm_magnitude': 0.20,
        'norm_consistency': 0.15,
        'cross_layer_consistency': 0.25,
        'avg_last_consistency': 0.15,
        'trajectory_consistency': 0.15,
        'layer_agreement': 0.10,
    }

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "auto",
        target_layers: Optional[List[int]] = None,
        token_weight: float = 0.5,
        activation_weight: float = 0.5,
    ):
        """
        Initialize the uncertainty estimator.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('auto', 'cuda', 'cpu')
            target_layers: List of layer indices to sample. If None, uses defaults.
            token_weight: Weight for token features (0-1)
            activation_weight: Weight for activation features (0-1)
        """
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        self.token_weight = token_weight
        self.activation_weight = activation_weight

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        self.device = next(self.model.parameters()).device
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

        # Set target layers
        if target_layers is not None:
            self.target_layers = [l for l in target_layers if l < self.num_layers]
        else:
            self.target_layers = get_default_layers(model_name, self.num_layers)

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(self.tokenizer, self.target_layers)

        print(f"Model: {self.num_layers} layers, {self.hidden_size} hidden dim")
        print(f"Device: {self.device}")
        print(f"Target layers ({len(self.target_layers)}): {self.target_layers}")
        print(f"Weights: token={self.token_weight:.0%}, activation={self.activation_weight:.0%}")

    @torch.no_grad()
    def estimate(
        self,
        question: str,
        temperature: float = 0.1,
        max_new_tokens: int = 50,
    ) -> Dict:
        """
        Generate answer and estimate correctness likelihood.

        Args:
            question: Question text
            temperature: Sampling temperature (0.0 for greedy)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with answer, confidence score, and prediction
        """
        # Format and tokenize
        prompt = format_prompt(question, self.model_name, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        # Generate answer
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs[0]
        answer_ids = generated_ids[input_length:]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        num_generated = len(answer_ids)

        if num_generated == 0:
            return {
                'question': question,
                'answer': answer,
                'confidence_score': 0.0,
                'uncertainty_score': 1.0,
                'predicted_correct': False,
                'num_tokens': 0,
            }

        # Forward pass to get hidden states and logits
        forward_outputs = self.model(
            generated_ids.unsqueeze(0),
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = forward_outputs.hidden_states
        logits = forward_outputs.logits[0, input_length:, :]
        probs = F.softmax(logits, dim=-1)

        gen_start = input_length
        gen_end = input_length + num_generated

        # Extract features
        token_features = self.feature_extractor.extract_token_features(answer_ids, probs)
        activation_features = self.feature_extractor.extract_activation_features(
            hidden_states, gen_start, gen_end
        )

        # Compute uncertainty scores
        token_uncertainty = self._compute_token_uncertainty(token_features)
        activation_uncertainty = self._compute_activation_uncertainty(activation_features)

        # Combine
        uncertainty_score = (
            self.token_weight * token_uncertainty +
            self.activation_weight * activation_uncertainty
        )

        # Clip to [0, 1]
        uncertainty_score = min(1.0, max(0.0, uncertainty_score))
        confidence_score = 1.0 - uncertainty_score

        # Predict correctness (threshold at 0.5)
        predicted_correct = confidence_score > 0.5

        return {
            'question': question,
            'answer': answer,
            'confidence_score': confidence_score,
            'uncertainty_score': uncertainty_score,
            'predicted_correct': predicted_correct,
            'num_tokens': num_generated,
            'token_uncertainty': token_uncertainty,
            'activation_uncertainty': activation_uncertainty,
            'features': {
                'token': token_features,
                'activation': activation_features,
            }
        }

    def estimate_batch(
        self,
        questions: List[str],
        temperature: float = 0.1,
        max_new_tokens: int = 50,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Process a batch of questions.

        Args:
            questions: List of question strings
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens per answer
            show_progress: Whether to show progress bar

        Returns:
            List of result dictionaries
        """
        results = []

        for idx, question in enumerate(questions):
            if show_progress:
                print(f"\rProcessing {idx+1}/{len(questions)}", end='', flush=True)

            try:
                result = self.estimate(question, temperature, max_new_tokens)
                results.append(result)
            except Exception as e:
                results.append({
                    'question': question,
                    'answer': '',
                    'confidence_score': 0.0,
                    'uncertainty_score': 1.0,
                    'predicted_correct': False,
                    'error': str(e),
                })

        if show_progress:
            print()

        return results

    def _compute_token_uncertainty(self, features: Dict[str, float]) -> float:
        """Compute uncertainty from token features."""
        components = []

        # Mean margin: high margin = low uncertainty
        margin_unc = 1.0 - min(1.0, max(0.0, (features['mean_margin'] - 0.2) / 0.6))
        components.append(self.TOKEN_WEIGHTS['mean_margin'] * margin_unc)

        # Min margin
        min_margin_unc = 1.0 - min(1.0, max(0.0, features['min_margin'] / 0.4))
        components.append(self.TOKEN_WEIGHTS['min_margin'] * min_margin_unc)

        # Percentage rank > 2
        components.append(self.TOKEN_WEIGHTS['pct_rank_gt_2'] * features['pct_rank_gt_2'])

        # Q1 confidence
        q1_unc = 1.0 - features['q1_confidence']
        components.append(self.TOKEN_WEIGHTS['q1_confidence'] * q1_unc)

        # Content confidence
        content_unc = 1.0 - features['content_mean_confidence']
        components.append(self.TOKEN_WEIGHTS['content_mean_confidence'] * content_unc)

        # Confidence trend
        trend_normalized = min(1.0, max(-1.0, features['confidence_trend'] * 30))
        trend_unc = (1.0 - trend_normalized) / 2
        components.append(self.TOKEN_WEIGHTS['confidence_trend'] * trend_unc)

        # Confidence Q4-Q1
        diff_normalized = min(1.0, max(-1.0, features['confidence_q4_minus_q1'] * 3))
        diff_unc = (1.0 - diff_normalized) / 2
        components.append(self.TOKEN_WEIGHTS['confidence_q4_minus_q1'] * diff_unc)

        # Mean confidence
        mean_unc = 1.0 - features['mean_confidence']
        components.append(self.TOKEN_WEIGHTS['mean_confidence'] * mean_unc)

        # Std confidence
        std_unc = min(1.0, features['std_confidence'] * 4)
        components.append(self.TOKEN_WEIGHTS['std_confidence'] * std_unc)

        # Last token confidence
        last_unc = 1.0 - features['last_token_confidence']
        components.append(self.TOKEN_WEIGHTS['last_token_confidence'] * last_unc)

        return sum(components)

    def _compute_activation_uncertainty(self, features: Dict[str, float]) -> float:
        """Compute uncertainty from activation features."""
        components = []

        # Norm magnitude (heuristic normalization)
        mean_norm = features['mean_norm']
        norm_unc = 1.0 - min(1.0, max(0.0, (mean_norm - 20) / 50))
        components.append(self.ACTIVATION_WEIGHTS['norm_magnitude'] * norm_unc)

        # Norm consistency
        norm_cons_unc = 1.0 - max(0, features['norm_consistency'])
        components.append(self.ACTIVATION_WEIGHTS['norm_consistency'] * norm_cons_unc)

        # Cross-layer consistency
        cross_layer = features['mean_cross_layer_consistency']
        cross_layer_unc = 1.0 - min(1.0, max(0.0, (cross_layer - 0.5) / 0.5))
        components.append(self.ACTIVATION_WEIGHTS['cross_layer_consistency'] * cross_layer_unc)

        # Avg-last consistency
        avg_last = features['mean_avg_last_consistency']
        avg_last_unc = 1.0 - max(0, avg_last)
        components.append(self.ACTIVATION_WEIGHTS['avg_last_consistency'] * avg_last_unc)

        # Trajectory consistency
        trajectory = features['mean_trajectory_consistency']
        trajectory_unc = 1.0 - max(0, trajectory)
        components.append(self.ACTIVATION_WEIGHTS['trajectory_consistency'] * trajectory_unc)

        # Layer agreement
        agreement = features['layer_agreement']
        agreement_unc = 1.0 - min(1.0, max(0.0, (agreement - 0.4) / 0.5))
        components.append(self.ACTIVATION_WEIGHTS['layer_agreement'] * agreement_unc)

        return sum(components)
