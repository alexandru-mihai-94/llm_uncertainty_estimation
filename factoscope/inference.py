"""
Inference Module for Factoscope

This module provides inference capabilities for the trained Factoscope model,
allowing confidence prediction on new questions.
"""

from typing import Dict, List

import numpy as np
import torch
import h5py
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import FactoscopeModel


class FactoscopeInference:
    """
    Inference engine for the Factoscope model.

    Provides methods to generate answers with confidence scores by using
    the trained factuality classifier on internal LLM states.
    """

    def __init__(
        self,
        model_path: str,
        factoscope_model_path: str,
        processed_data_path: str,
        device: str = 'cpu'
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to the LLM (e.g., Meta-Llama-3-8B)
            factoscope_model_path: Path to trained factoscope model (.pt file)
            processed_data_path: Path to processed training data (for support set and normalization)
            device: Device to use ('cuda' or 'cpu')
        """
        print("Initializing Factoscope Inference Engine...")
        self.device = device

        # Load LLM
        print(f"Loading LLM from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto' if device == 'cuda' else None,
            low_cpu_mem_usage=True
        )

        if device == 'cpu':
            self.llm = self.llm.to(device)

        self.llm.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load processed data to get normalization params and support set
        print(f"Loading processed data from: {processed_data_path}")
        with h5py.File(processed_data_path, 'r') as f:
            # Get normalization parameters
            self.mean = f.attrs['mean']
            self.std = f.attrs['std']

            # Load support set (use subset for efficiency)
            hidden_states = f['hidden_states'][:]
            ranks = f['ranks'][:]
            probs = f['topk_probs'][:]
            labels = f['labels'][:]

            # Create balanced support set
            correct_indices = np.where(labels == 1)[0]
            false_indices = np.where(labels == 0)[0]

            # Sample equal numbers
            n_support = min(50, len(correct_indices), len(false_indices))
            correct_sample = np.random.choice(correct_indices, n_support, replace=False)
            false_sample = np.random.choice(false_indices, n_support, replace=False)
            support_indices = np.concatenate([correct_sample, false_sample])

            self.support_hidden = torch.from_numpy(hidden_states[support_indices]).float()
            self.support_ranks = torch.from_numpy(ranks[support_indices]).float()
            self.support_probs = torch.from_numpy(probs[support_indices]).float()
            self.support_labels = torch.from_numpy(labels[support_indices]).long()

            num_layers = hidden_states.shape[1]
            hidden_dim = hidden_states.shape[2]

        # Load factoscope model
        print(f"Loading factoscope model from: {factoscope_model_path}")
        self.factoscope = FactoscopeModel(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            emb_dim=32,
            final_dim=64
        )
        self.factoscope.load_state_dict(torch.load(factoscope_model_path, map_location=device))
        self.factoscope.to(device)
        self.factoscope.eval()

        # Compute support set embeddings
        print("Computing support set embeddings...")
        with torch.no_grad():
            self.support_embeddings = self.factoscope(
                self.support_hidden.to(device),
                self.support_ranks.to(device),
                self.support_probs.to(device)
            )

        print(f"Initialization complete!")
        print(f"  Support set size: {len(self.support_labels)}")
        print(f"  Correct examples: {(self.support_labels == 1).sum().item()}")
        print(f"  False examples: {(self.support_labels == 0).sum().item()}")

    def generate_and_collect(self, prompt: str) -> Dict:
        """
        Generate answer and collect internal states.

        Args:
            prompt: Input question/prompt

        Returns:
            Dictionary containing generated text, hidden states, probabilities, and rank
        """
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=10,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Extract generated text
        generated_ids = outputs.sequences[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if len(generated_ids) > 0:
            first_token_id = generated_ids[0].item()
            first_token = self.tokenizer.decode([first_token_id]).strip()
        else:
            first_token = ""

        # Collect hidden states
        if hasattr(outputs, 'hidden_states') and len(outputs.hidden_states) > 0:
            first_step_hidden = outputs.hidden_states[0]
            layer_activations = []
            for layer_hidden in first_step_hidden:
                last_pos_hidden = layer_hidden[0, -1, :].cpu().float()
                layer_activations.append(last_pos_hidden)
            hidden_states = torch.stack(layer_activations, dim=0)
        else:
            hidden_states = None

        # Get logits and probabilities
        with torch.no_grad():
            model_outputs = self.llm(**inputs)
            logits = model_outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

            topk_probs, topk_ids = torch.topk(probs, k=10)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            token_ranks = {idx.item(): rank for rank, idx in enumerate(sorted_indices)}

        return {
            'generated_text': generated_text,
            'first_token': first_token,
            'hidden_states': hidden_states,
            'topk_probs': topk_probs.cpu(),
            'rank': token_ranks.get(first_token_id, len(probs)) if len(generated_ids) > 0 else 0
        }

    def transform_rank(self, rank: float) -> float:
        """
        Transform rank using same method as training.

        Args:
            rank: Raw rank value

        Returns:
            Transformed rank value
        """
        a = -1
        return 1 / (a * (rank - 1) + 1 + 1e-7)

    def predict_confidence(self, prompt: str) -> Dict:
        """
        Predict confidence for a question.

        Args:
            prompt: Input question

        Returns:
            Dictionary with answer, confidence score, and detailed metrics:
                - prompt: The input prompt
                - answer: Generated answer text
                - first_token: First generated token
                - confidence: Weighted confidence score (0-1)
                - simple_confidence: Simple voting confidence (0-1)
                - prediction: "correct" or "false"
                - nearest_correct_distance: Distance to nearest correct example
                - nearest_false_distance: Distance to nearest false example
                - distance_ratio: Ratio of correct to false distance
                - top_k_neighbors: Count of correct/false in k-nearest neighbors
                - rank: Token rank in probability distribution
                - top_prob: Highest token probability
        """
        # Generate and collect features
        result = self.generate_and_collect(prompt)

        if result['hidden_states'] is None:
            return {
                'prompt': prompt,
                'answer': result['generated_text'],
                'confidence': 0.0,
                'error': 'Failed to collect hidden states'
            }

        # Normalize features
        hidden = result['hidden_states'].numpy()
        hidden = (hidden - self.mean) / (self.std + 1e-7)
        hidden = torch.from_numpy(hidden).float().unsqueeze(0).to(self.device)

        rank = self.transform_rank(result['rank'])
        rank = torch.tensor([rank]).float().to(self.device)

        probs = result['topk_probs'].unsqueeze(0).to(self.device)

        # Get embedding
        with torch.no_grad():
            embedding = self.factoscope(hidden, rank, probs)

        # Calculate distances to support set
        distances = torch.cdist(embedding, self.support_embeddings, p=2)
        distances = distances.squeeze(0)  # (support_size,)

        # Find k nearest neighbors
        k = 10
        topk_distances, topk_indices = torch.topk(distances, k, largest=False)

        # Get labels of nearest neighbors
        neighbor_labels = self.support_labels[topk_indices].cpu().numpy()
        neighbor_distances = topk_distances.cpu().numpy()

        # Calculate confidence as weighted vote
        # Closer neighbors have more weight
        weights = 1 / (neighbor_distances + 1e-6)
        weighted_correct = np.sum(weights[neighbor_labels == 1])
        weighted_false = np.sum(weights[neighbor_labels == 0])

        confidence = weighted_correct / (weighted_correct + weighted_false)

        # Alternative: simple vote
        simple_confidence = np.mean(neighbor_labels)

        # Get nearest correct and false examples
        correct_distances = distances[self.support_labels == 1]
        false_distances = distances[self.support_labels == 0]

        min_correct_dist = correct_distances.min().item() if len(correct_distances) > 0 else float('inf')
        min_false_dist = false_distances.min().item() if len(false_distances) > 0 else float('inf')

        # Classification based on nearest neighbor
        prediction = "correct" if min_correct_dist < min_false_dist else "false"

        return {
            'prompt': prompt,
            'answer': result['generated_text'],
            'first_token': result['first_token'],
            'confidence': float(confidence),
            'simple_confidence': float(simple_confidence),
            'prediction': prediction,
            'nearest_correct_distance': float(min_correct_dist),
            'nearest_false_distance': float(min_false_dist),
            'distance_ratio': float(min_correct_dist / (min_false_dist + 1e-6)),
            'top_k_neighbors': {
                'correct': int(np.sum(neighbor_labels == 1)),
                'false': int(np.sum(neighbor_labels == 0))
            },
            'rank': result['rank'],
            'top_prob': float(result['topk_probs'][0])
        }

    def batch_predict(self, prompts: List[str]) -> List[Dict]:
        """
        Predict confidence for multiple prompts.

        Args:
            prompts: List of input questions

        Returns:
            List of prediction dictionaries (one per prompt)
        """
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\nProcessing {i+1}/{len(prompts)}: {prompt[:60]}...")
            result = self.predict_confidence(prompt)
            print(f"\n Results: {result}...")
            results.append(result)
        return results
