"""
Data Collection Module for Factoscope

This module handles collecting internal states from LLM when answering factual questions.
It extracts hidden states, probabilities, and ranks from the model's generation process.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import h5py
from transformers import AutoModelForCausalLM, AutoTokenizer


class FactDataCollector:
    """
    Collects internal states from LLM when answering factual questions.

    This class loads a language model and processes factual questions, collecting
    hidden states, token probabilities, and other internal features that can be
    used to train a factuality classifier.
    """

    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize the data collector.

        Args:
            model_path: Path to the model directory (e.g., Meta-Llama-3-8B)
            device: Device to run on ('cuda' or 'cpu')
        """
        print(f"\nInitializing Factoscope Data Collector")
        print(f"Loading model from: {model_path}")

        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None,
            low_cpu_mem_usage=True
        )

        if self.device == 'cpu':
            self.model = self.model.to(self.device)

        self.model.eval()

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Words to classify as "unrelative" (articles, pronouns, etc.)
        self.unrelative_tokens = {
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them',
            'to', 'of', 'not', 'at', 'in', 'on',
            'when', 'where', 'what', 'who', 'how', 'why'
        }

    def generate_and_collect(self, prompt: str, max_new_tokens: int = 10) -> Dict:
        """
        Generate response and collect internal states.

        Args:
            prompt: Input prompt to process
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary containing:
                - generated_text: Full generated response
                - first_token: First generated token
                - first_token_id: ID of first generated token
                - hidden_states: Hidden states from all layers (num_layers, hidden_dim)
                - topk_ids: IDs of top-k predicted tokens
                - topk_probs: Probabilities of top-k tokens
                - first_token_rank: Rank of first generated token
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_length = inputs['input_ids'].shape[1]

        # Generate with output_hidden_states
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,  # Greedy decoding
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Get generated tokens
        generated_ids = outputs.sequences[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract first generated token
        if len(generated_ids) > 0:
            first_token_id = generated_ids[0].item()
            first_token = self.tokenizer.decode([first_token_id]).strip()
        else:
            first_token = ""

        # Collect hidden states from the last input position
        # outputs.hidden_states is a tuple of tuples: (generation_step, layer, batch, seq, hidden)
        if hasattr(outputs, 'hidden_states') and len(outputs.hidden_states) > 0:
            # Get hidden states from first generation step
            first_step_hidden = outputs.hidden_states[0]  # Tuple of layers

            # Extract activations from each layer's last position
            layer_activations = []
            for layer_hidden in first_step_hidden:
                # layer_hidden shape: (batch, seq_len, hidden_dim)
                last_pos_hidden = layer_hidden[0, -1, :].cpu().float()
                layer_activations.append(last_pos_hidden)

            # Stack all layers: (num_layers, hidden_dim)
            hidden_states = torch.stack(layer_activations, dim=0)
        else:
            hidden_states = None

        # Get logits and calculate probabilities for top-k tokens
        with torch.no_grad():
            # Forward pass to get logits at last position
            model_outputs = self.model(**inputs)
            logits = model_outputs.logits[0, -1, :]  # Last position logits
            probs = torch.softmax(logits, dim=-1)

            # Get top-k tokens and their probabilities
            topk_probs, topk_ids = torch.topk(probs, k=10)

            # Calculate rank of each token (for ranking metric)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            token_ranks = {idx.item(): rank for rank, idx in enumerate(sorted_indices)}

        return {
            'generated_text': generated_text,
            'first_token': first_token,
            'first_token_id': first_token_id if len(generated_ids) > 0 else None,
            'hidden_states': hidden_states,  # Shape: (num_layers, hidden_dim)
            'topk_ids': topk_ids.cpu(),
            'topk_probs': topk_probs.cpu(),
            'first_token_rank': token_ranks.get(first_token_id, len(probs)) if len(generated_ids) > 0 else 0
        }

    def process_dataset(
        self,
        dataset_path: str,
        output_dir: str,
        max_samples: Optional[int] = 1000
    ) -> tuple:
        """
        Process a dataset file and collect internal states.

        Args:
            dataset_path: Path to JSON dataset file containing prompts and answers
            output_dir: Directory to save collected features
            max_samples: Maximum number of samples to process (None for all)

        Returns:
            Tuple of (num_correct, num_false, num_unrelative)
        """
        # Load dataset
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        dataset_name = Path(dataset_path).stem.replace('_dataset', '')
        print(f"\nProcessing dataset: {dataset_name}")
        print(f"Total samples: {len(data)}, Processing: {min(len(data), max_samples)}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Storage for categorized samples
        correct_samples = []
        false_samples = []
        unrelative_samples = []

        # Process samples
        for i, entry in enumerate(data[:max_samples]):
            if i < 5 or i % 10 == 0:  # Show first 5, then every 10
                print(f"  Processing sample {i+1}/{min(len(data), max_samples)}...", flush=True)

            prompt = entry['prompt']
            answer = entry['answer']

            # Handle multi-word answers
            if isinstance(answer, str):
                target = answer.split()[0] if len(answer.split()) > 1 else answer
            else:
                continue

            # Generate and collect states
            result = self.generate_and_collect(prompt, max_new_tokens=10)

            if result['hidden_states'] is None:
                continue

            # Clean tokens
            first_token = result['first_token'].lower().strip('.,!?;:\"')
            target = target.lower().strip('.,!?;:\"')

            # Prepare data entry
            data_entry = {
                'index': entry.get('index', i),
                'prompt': prompt,
                'answer': target,
                'generated': first_token,
                'hidden_states': result['hidden_states'].numpy(),  # (num_layers, hidden_dim)
                'topk_ids': result['topk_ids'].numpy(),
                'topk_probs': result['topk_probs'].numpy(),
                'rank': result['first_token_rank']
            }

            # Categorize
            if first_token == target:
                correct_samples.append(data_entry)
            elif first_token in self.unrelative_tokens:
                unrelative_samples.append(data_entry)
            else:
                false_samples.append(data_entry)

        print(f"\nResults:")
        print(f"  Correct: {len(correct_samples)}")
        print(f"  False: {len(false_samples)}")
        print(f"  Unrelative: {len(unrelative_samples)}")

        # Save to HDF5
        self._save_to_hdf5(correct_samples, os.path.join(output_dir, 'correct_data.h5'))
        self._save_to_hdf5(false_samples, os.path.join(output_dir, 'false_data.h5'))
        self._save_to_hdf5(unrelative_samples, os.path.join(output_dir, 'unrelative_data.h5'))

        # Save metadata JSON
        metadata = {
            'correct': [{'index': s['index'], 'prompt': s['prompt'],
                        'answer': s['answer'], 'generated': s['generated']}
                       for s in correct_samples],
            'false': [{'index': s['index'], 'prompt': s['prompt'],
                      'answer': s['answer'], 'generated': s['generated']}
                     for s in false_samples],
            'unrelative': [{'index': s['index'], 'prompt': s['prompt'],
                           'answer': s['answer'], 'generated': s['generated']}
                          for s in unrelative_samples]
        }

        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        return len(correct_samples), len(false_samples), len(unrelative_samples)

    def _save_to_hdf5(self, samples: List[Dict], filepath: str) -> None:
        """
        Save samples to HDF5 file.

        Args:
            samples: List of sample dictionaries
            filepath: Path to save HDF5 file
        """
        if len(samples) == 0:
            return

        with h5py.File(filepath, 'w') as f:
            # Stack hidden states: (num_samples, num_layers, hidden_dim)
            hidden_states = np.stack([s['hidden_states'] for s in samples], axis=0)
            f.create_dataset('hidden_states', data=hidden_states, dtype=np.float32)

            # Ranks: (num_samples,)
            ranks = np.array([s['rank'] for s in samples], dtype=np.int32)
            f.create_dataset('ranks', data=ranks)

            # Top-k IDs: (num_samples, 10)
            topk_ids = np.stack([s['topk_ids'] for s in samples], axis=0)
            f.create_dataset('topk_ids', data=topk_ids, dtype=np.int32)

            # Top-k probs: (num_samples, 10)
            topk_probs = np.stack([s['topk_probs'] for s in samples], axis=0)
            f.create_dataset('topk_probs', data=topk_probs, dtype=np.float32)
