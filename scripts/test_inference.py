#!/usr/bin/env python3
"""
Factoscope Inference - Test trained model on new questions
Provides uncertainty/confidence scores for LLM responses
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings('ignore')


# ============================================================================
# Load model architectures (must match training)
# ============================================================================

class HiddenStateEncoder(nn.Module):
    """Encodes hidden states using 1D convolutions"""

    def __init__(self, num_layers, hidden_dim, emb_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class RankEncoder(nn.Module):
    """Encodes rank using simple MLP"""

    def __init__(self, emb_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, emb_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ProbEncoder(nn.Module):
    """Encodes top-k probabilities"""

    def __init__(self, emb_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, emb_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FactoscopeModel(nn.Module):
    """Complete Factoscope model combining all features"""

    def __init__(self, num_layers, hidden_dim, emb_dim=32, final_dim=64):
        super().__init__()
        self.hidden_encoder = HiddenStateEncoder(num_layers, hidden_dim, emb_dim)
        self.rank_encoder = RankEncoder(emb_dim)
        self.prob_encoder = ProbEncoder(emb_dim)

        self.combiner = nn.Sequential(
            nn.Linear(emb_dim * 3, final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, final_dim)
        )

    def forward(self, hidden, rank, prob):
        h_emb = self.hidden_encoder(hidden)
        r_emb = self.rank_encoder(rank)
        p_emb = self.prob_encoder(prob)

        combined = torch.cat([h_emb, r_emb, p_emb], dim=1)
        output = self.combiner(combined)
        output = F.normalize(output, p=2, dim=1)
        return output


# ============================================================================
# Inference Engine
# ============================================================================

class FactoscopeInference:
    """Inference engine for factoscope model"""

    def __init__(self, model_path: str, factoscope_model_path: str,
                 processed_data_path: str, device: str = 'cpu'):
        """
        Initialize inference engine

        Args:
            model_path: Path to LLM (Meta-Llama-3-8B)
            factoscope_model_path: Path to trained factoscope model
            processed_data_path: Path to processed training data (for support set)
            device: Device to use
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

        print(f"✓ Initialization complete!")
        print(f"  Support set size: {len(self.support_labels)}")
        print(f"  Correct examples: {(self.support_labels == 1).sum().item()}")
        print(f"  False examples: {(self.support_labels == 0).sum().item()}")

    def generate_and_collect(self, prompt: str) -> Dict:
        """Generate answer and collect internal states"""
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
        """Transform rank using same method as training"""
        a = -1
        return 1 / (a * (rank - 1) + 1 + 1e-7)

    def predict_confidence(self, prompt: str) -> Dict:
        """
        Predict confidence for a question

        Args:
            prompt: Input question

        Returns:
            Dictionary with answer, confidence, and details
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
        """Predict confidence for multiple prompts"""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\nProcessing {i+1}/{len(prompts)}: {prompt[:60]}...")
            result = self.predict_confidence(prompt)
            print(f"\n Results: {result}...")
            results.append(result)
        return results


# ============================================================================
# Main Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Factoscope Inference')
    parser.add_argument('--model_path', type=str, default='./models/Meta-Llama-3-8B',
                       help='Path to LLM')
    parser.add_argument('--factoscope_model', type=str,
                       default='./factoscope_output/best_factoscope_model.pt',
                       help='Path to trained factoscope model')
    parser.add_argument('--processed_data', type=str,
                       default='./factoscope_output/processed_data.h5',
                       help='Path to processed training data')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    parser.add_argument('--questions', type=str, nargs='+',
                       help='Questions to test')
    parser.add_argument('--questions_file', type=str,
                       help='JSON file with questions')
    parser.add_argument('--output', type=str,
                       help='Output JSON file for results')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')

    args = parser.parse_args()

    # Initialize inference engine
    engine = FactoscopeInference(
        model_path=args.model_path,
        factoscope_model_path=args.factoscope_model,
        processed_data_path=args.processed_data,
        device=args.device
    )

    # Collect questions
    questions = []

    if args.questions:
        questions.extend(args.questions)

    if args.questions_file:
        with open(args.questions_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                questions.extend([q['prompt'] if isinstance(q, dict) else q for q in data])
            else:
                questions.extend(data.get('questions', []))

    # Interactive mode
    if args.interactive or (not questions and not args.questions_file):
        print("\n" + "="*70)
        print("FACTOSCOPE INTERACTIVE MODE")
        print("="*70)
        print("Enter questions to test (or 'quit' to exit)")
        print()

        while True:
            try:
                prompt = input("Question: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                if not prompt:
                    continue

                result = engine.predict_confidence(prompt)

                print("\n" + "-"*70)
                print(f"Answer: {result['answer']}")
                print(f"First Token: {result['first_token']}")
                print(f"Confidence: {result['confidence']:.2%} (weighted)")
                print(f"Simple Confidence: {result['simple_confidence']:.2%}")
                print(f"Prediction: {result['prediction'].upper()}")
                print(f"Nearest Correct Distance: {result['nearest_correct_distance']:.4f}")
                print(f"Nearest False Distance: {result['nearest_false_distance']:.4f}")
                print(f"Distance Ratio: {result['distance_ratio']:.4f} ({'favors correct' if result['distance_ratio'] < 1 else 'favors false'})")
                print(f"Top-{sum(result['top_k_neighbors'].values())} Neighbors: "
                      f"{result['top_k_neighbors']['correct']} correct, {result['top_k_neighbors']['false']} false")
                print(f"Token Rank: {result['rank']}, Top Probability: {result['top_prob']:.4f}")
                print("-"*70 + "\n")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

    # Batch mode
    elif questions:
        print(f"\nProcessing {len(questions)} questions...")
        results = engine.batch_predict(questions)

        # Print summary
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['prompt'][:60]}")
            print(f"   Answer: {result['answer'][:40]}")
            print(f"   Confidence: {result['confidence']:.2%} | Prediction: {result['prediction']}")

        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {args.output}")

    print("\n✓ Done!\n")


if __name__ == '__main__':
    main()
