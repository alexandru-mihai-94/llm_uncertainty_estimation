"""
Feature extraction for LLM uncertainty estimation.

Extracts token-level statistics and activation-based features
from language model outputs.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils import is_content_word, compute_cosine_similarity


class FeatureExtractor:
    """Extract features from LLM outputs for uncertainty estimation."""

    def __init__(self, tokenizer, target_layers: List[int]):
        """
        Initialize feature extractor.

        Args:
            tokenizer: HuggingFace tokenizer
            target_layers: List of layer indices to extract activations from
        """
        self.tokenizer = tokenizer
        self.target_layers = target_layers

    def extract_token_features(
        self,
        answer_ids: torch.Tensor,
        probs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Extract token-level statistical features.

        Args:
            answer_ids: Generated token IDs [seq_len]
            probs: Token probabilities [seq_len, vocab_size]

        Returns:
            Dictionary of token-level features
        """
        num_tokens = len(answer_ids)
        if num_tokens == 0:
            return self._empty_token_features()

        confidences = []
        margins = []
        ranks = []
        content_confidences = []

        for t in range(num_tokens):
            token_id = answer_ids[t].item()
            token_text = self.tokenizer.decode([token_id])

            token_probs = probs[t]

            # Confidence: probability of selected token
            confidence = token_probs[token_id].item()
            confidences.append(confidence)

            # Margin: difference between top-1 and top-2
            top2_probs, _ = torch.topk(token_probs, min(2, len(token_probs)))
            margin = (top2_probs[0] - top2_probs[1]).item() if len(top2_probs) > 1 else 1.0
            margins.append(margin)

            # Rank: position of selected token
            sorted_indices = torch.argsort(token_probs, descending=True)
            rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)

            # Content word confidence
            if is_content_word(token_text):
                content_confidences.append(confidence)

        # Quartile boundaries
        q1_end = max(1, num_tokens // 4)
        q4_start = max(1, 3 * num_tokens // 4)

        features = {
            # Basic statistics
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences) if len(confidences) > 1 else 0,
            'min_confidence': np.min(confidences),
            'last_token_confidence': confidences[-1],

            # Margin statistics
            'mean_margin': np.mean(margins),
            'min_margin': np.min(margins),
            'std_margin': np.std(margins) if len(margins) > 1 else 0,

            # Ranking
            'pct_rank_gt_2': np.mean([r > 2 for r in ranks]),

            # Quartile analysis
            'q1_confidence': np.mean(confidences[:q1_end]),
            'q4_confidence': np.mean(confidences[q4_start:]),
            'confidence_q4_minus_q1': np.mean(confidences[q4_start:]) - np.mean(confidences[:q1_end]),

            # Trend
            'confidence_trend': np.polyfit(range(num_tokens), confidences, 1)[0] if num_tokens > 2 else 0,

            # Content words
            'content_mean_confidence': np.mean(content_confidences) if content_confidences else np.mean(confidences),
        }

        return features

    def extract_activation_features(
        self,
        hidden_states: Tuple,
        gen_start: int,
        gen_end: int
    ) -> Dict[str, float]:
        """
        Extract activation-based features from hidden states.

        Args:
            hidden_states: Tuple of layer hidden states
            gen_start: Start index of generated tokens
            gen_end: End index of generated tokens

        Returns:
            Dictionary of activation features
        """
        if gen_end <= gen_start:
            return self._empty_activation_features()

        layer_data = {}
        all_norms = []

        # Extract data from each target layer
        for layer_idx in self.target_layers:
            if layer_idx + 1 >= len(hidden_states):
                continue

            layer_hidden = hidden_states[layer_idx + 1]
            gen_hidden = layer_hidden[0, gen_start:gen_end, :].float()

            # Averaged vector across tokens
            avg_vector = gen_hidden.mean(dim=0).cpu().numpy()
            last_vector = gen_hidden[-1].cpu().numpy()
            per_token = gen_hidden.cpu().numpy()

            # Norms
            avg_norm = np.linalg.norm(avg_vector)
            last_norm = np.linalg.norm(last_vector)
            all_norms.extend([avg_norm, last_norm])

            # Avg-last consistency
            avg_last_sim = compute_cosine_similarity(avg_vector, last_vector)

            # Sparsity
            threshold = np.std(avg_vector)
            sparsity = np.mean(np.abs(avg_vector) > threshold) if threshold > 0 else 0.5

            # Trajectory consistency
            if len(per_token) > 1:
                consecutive_sims = []
                for i in range(len(per_token) - 1):
                    sim = compute_cosine_similarity(per_token[i], per_token[i + 1])
                    consecutive_sims.append(sim)
                trajectory_consistency = np.mean(consecutive_sims) if consecutive_sims else 1.0
            else:
                trajectory_consistency = 1.0

            layer_data[layer_idx] = {
                'avg_vector': avg_vector,
                'last_vector': last_vector,
                'avg_norm': avg_norm,
                'last_norm': last_norm,
                'avg_last_sim': avg_last_sim,
                'sparsity': sparsity,
                'trajectory_consistency': trajectory_consistency,
            }

        if not layer_data:
            return self._empty_activation_features()

        # Aggregate across layers
        layer_indices = sorted(layer_data.keys())

        # Norm statistics
        norms = [layer_data[l]['avg_norm'] for l in layer_indices]
        mean_norm = np.mean(norms)
        std_norm = np.std(norms) if len(norms) > 1 else 0
        norm_consistency = 1.0 - (std_norm / (mean_norm + 1e-6))
        norm_consistency = max(0, min(1, norm_consistency))

        # Cross-layer consistency
        cross_layer_sims = []
        for i in range(len(layer_indices) - 1):
            l1, l2 = layer_indices[i], layer_indices[i + 1]
            v1, v2 = layer_data[l1]['avg_vector'], layer_data[l2]['avg_vector']
            sim = compute_cosine_similarity(v1, v2)
            cross_layer_sims.append(sim)

        mean_cross_layer = np.mean(cross_layer_sims) if cross_layer_sims else 0.5

        # Avg-last consistency
        avg_last_sims = [layer_data[l]['avg_last_sim'] for l in layer_indices]
        mean_avg_last = np.mean(avg_last_sims)

        # Trajectory consistency
        trajectory_sims = [layer_data[l]['trajectory_consistency'] for l in layer_indices]
        mean_trajectory = np.mean(trajectory_sims)

        # Sparsity
        sparsities = [layer_data[l]['sparsity'] for l in layer_indices]
        mean_sparsity = np.mean(sparsities)

        # Layer agreement (all pairwise similarities)
        if len(layer_indices) >= 3:
            all_pairs_sims = []
            for i, l1 in enumerate(layer_indices):
                for l2 in layer_indices[i+1:]:
                    v1, v2 = layer_data[l1]['avg_vector'], layer_data[l2]['avg_vector']
                    sim = compute_cosine_similarity(v1, v2)
                    all_pairs_sims.append(sim)
            layer_agreement = np.mean(all_pairs_sims) if all_pairs_sims else 0.5
        else:
            layer_agreement = mean_cross_layer

        features = {
            'mean_norm': mean_norm,
            'std_norm': std_norm,
            'norm_consistency': norm_consistency,
            'mean_cross_layer_consistency': mean_cross_layer,
            'mean_avg_last_consistency': mean_avg_last,
            'mean_trajectory_consistency': mean_trajectory,
            'mean_sparsity': mean_sparsity,
            'layer_agreement': layer_agreement,
        }

        return features

    def _empty_token_features(self) -> Dict[str, float]:
        """Return empty token features."""
        return {
            'mean_confidence': 0.0,
            'std_confidence': 0.0,
            'min_confidence': 0.0,
            'last_token_confidence': 0.0,
            'mean_margin': 0.0,
            'min_margin': 0.0,
            'std_margin': 0.0,
            'pct_rank_gt_2': 1.0,
            'q1_confidence': 0.0,
            'q4_confidence': 0.0,
            'confidence_q4_minus_q1': 0.0,
            'confidence_trend': 0.0,
            'content_mean_confidence': 0.0,
        }

    def _empty_activation_features(self) -> Dict[str, float]:
        """Return empty activation features."""
        return {
            'mean_norm': 0.0,
            'std_norm': 0.0,
            'norm_consistency': 0.0,
            'mean_cross_layer_consistency': 0.0,
            'mean_avg_last_consistency': 0.0,
            'mean_trajectory_consistency': 0.0,
            'mean_sparsity': 0.0,
            'layer_agreement': 0.0,
        }
