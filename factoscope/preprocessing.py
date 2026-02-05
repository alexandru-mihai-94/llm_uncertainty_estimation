"""
Data Preprocessing Module for Factoscope

This module handles normalization and preparation of collected data for training.
It balances datasets, normalizes features, and transforms ranks.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import h5py


class FactDataPreprocessor:
    """
    Preprocesses collected data for training.

    This class handles loading, balancing, normalizing, and preparing data
    for the factuality classifier training process.
    """

    def __init__(self, feature_dir: str):
        """
        Initialize preprocessor.

        Args:
            feature_dir: Directory containing collected feature files
        """
        self.feature_dir = Path(feature_dir)

    def load_and_balance(
        self,
        correct_path: str,
        false_path: str,
        unrelative_path: str
    ) -> Tuple[np.ndarray, ...]:
        """
        Load and balance data from HDF5 files.

        Combines false and unrelative samples as "incorrect" and balances with
        correct samples by randomly sampling to match the smaller class size.

        Args:
            correct_path: Path to correct data HDF5 file
            false_path: Path to false data HDF5 file
            unrelative_path: Path to unrelative data HDF5 file

        Returns:
            Tuple of (correct_hidden, correct_ranks, correct_topk_probs,
                     false_hidden, false_ranks, false_topk_probs)
        """
        # Load correct data
        with h5py.File(correct_path, 'r') as f:
            correct_hidden = f['hidden_states'][:]
            correct_ranks = f['ranks'][:]
            correct_topk_probs = f['topk_probs'][:]

        # Load false data
        with h5py.File(false_path, 'r') as f:
            false_hidden = f['hidden_states'][:]
            false_ranks = f['ranks'][:]
            false_topk_probs = f['topk_probs'][:]

        # Load unrelative data if exists
        if os.path.exists(unrelative_path):
            with h5py.File(unrelative_path, 'r') as f:
                unrel_hidden = f['hidden_states'][:]
                unrel_ranks = f['ranks'][:]
                unrel_topk_probs = f['topk_probs'][:]
        else:
            unrel_hidden = np.array([])
            unrel_ranks = np.array([])
            unrel_topk_probs = np.array([])

        # Balance: combine false and unrelative as "incorrect"
        num_correct = len(correct_hidden)

        # Combine false and unrelative
        if len(unrel_hidden) > 0:
            false_hidden = np.concatenate([false_hidden, unrel_hidden], axis=0)
            false_ranks = np.concatenate([false_ranks, unrel_ranks], axis=0)
            false_topk_probs = np.concatenate([false_topk_probs, unrel_topk_probs], axis=0)

        # Sample to balance
        if len(false_hidden) > num_correct:
            indices = np.random.choice(len(false_hidden), num_correct, replace=False)
            false_hidden = false_hidden[indices]
            false_ranks = false_ranks[indices]
            false_topk_probs = false_topk_probs[indices]

        return (correct_hidden, correct_ranks, correct_topk_probs,
                false_hidden, false_ranks, false_topk_probs)

    def normalize_data(
        self,
        data: np.ndarray,
        mean: Optional[float] = None,
        std: Optional[float] = None
    ) -> Tuple[np.ndarray, float, float]:
        """
        Normalize data using z-score normalization.

        Args:
            data: Data array to normalize
            mean: Pre-computed mean (None to compute from data)
            std: Pre-computed standard deviation (None to compute from data)

        Returns:
            Tuple of (normalized_data, mean, std)
        """
        if mean is None:
            mean = np.mean(data)
        if std is None:
            std = np.std(data)

        normalized = (data - mean) / (std + 1e-7)
        return normalized, mean, std

    def transform_ranks(self, ranks: np.ndarray) -> np.ndarray:
        """
        Transform rank data using inverse transformation.

        This emphasizes the importance of top ranks by applying an inverse
        transformation that maps ranks to the [0, 1] range.

        Args:
            ranks: Array of rank values

        Returns:
            Transformed rank values
        """
        # Transform ranks to [0, 1] range with emphasis on top ranks
        a = -1
        transformed = 1 / (a * (ranks - 1) + 1 + 1e-7)
        return transformed

    def prepare_training_data(
        self,
        dataset_paths: List[str],
        output_file: str
    ) -> Dict:
        """
        Prepare data for training from multiple datasets.

        Loads data from multiple dataset directories, combines them, balances
        classes, normalizes features, and saves the processed data.

        Args:
            dataset_paths: List of paths to dataset directories containing HDF5 files
            output_file: Path to save the processed data HDF5 file

        Returns:
            Dictionary with statistics:
                - num_samples: Total number of samples
                - num_correct: Number of correct samples
                - num_false: Number of false samples
                - mean: Mean used for normalization
                - std: Standard deviation used for normalization
        """
        print("\n" + "="*70)
        print("PREPARING TRAINING DATA")
        print("="*70)

        all_correct_hidden = []
        all_correct_ranks = []
        all_correct_probs = []
        all_false_hidden = []
        all_false_ranks = []
        all_false_probs = []

        # Collect data from all datasets
        for dataset_dir in dataset_paths:
            correct_path = os.path.join(dataset_dir, 'correct_data.h5')
            false_path = os.path.join(dataset_dir, 'false_data.h5')
            unrelative_path = os.path.join(dataset_dir, 'unrelative_data.h5')

            if not os.path.exists(correct_path) or not os.path.exists(false_path):
                print(f"Skipping {dataset_dir} - missing files")
                continue

            print(f"\nLoading from: {dataset_dir}")

            correct_h, correct_r, correct_p, false_h, false_r, false_p = \
                self.load_and_balance(correct_path, false_path, unrelative_path)

            all_correct_hidden.append(correct_h)
            all_correct_ranks.append(correct_r)
            all_correct_probs.append(correct_p)
            all_false_hidden.append(false_h)
            all_false_ranks.append(false_r)
            all_false_probs.append(false_p)

            print(f"  Correct: {len(correct_h)}, False: {len(false_h)}")

        # Concatenate all data
        correct_hidden = np.concatenate(all_correct_hidden, axis=0)
        correct_ranks = np.concatenate(all_correct_ranks, axis=0)
        correct_probs = np.concatenate(all_correct_probs, axis=0)
        false_hidden = np.concatenate(all_false_hidden, axis=0)
        false_ranks = np.concatenate(all_false_ranks, axis=0)
        false_probs = np.concatenate(all_false_probs, axis=0)

        print(f"\nTotal samples - Correct: {len(correct_hidden)}, False: {len(false_hidden)}")

        # Combine and create labels
        all_hidden = np.concatenate([correct_hidden, false_hidden], axis=0)
        all_ranks = np.concatenate([correct_ranks, false_ranks], axis=0)
        all_probs = np.concatenate([correct_probs, false_probs], axis=0)
        all_labels = np.concatenate([
            np.ones(len(correct_hidden)),
            np.zeros(len(false_hidden))
        ], axis=0)

        # Normalize hidden states
        print("\nNormalizing data...")
        all_hidden, mean, std = self.normalize_data(all_hidden)

        # Transform ranks
        all_ranks = self.transform_ranks(all_ranks)

        # Save processed data
        print(f"Saving to: {output_file}")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('hidden_states', data=all_hidden, dtype=np.float32)
            f.create_dataset('ranks', data=all_ranks, dtype=np.float32)
            f.create_dataset('topk_probs', data=all_probs, dtype=np.float32)
            f.create_dataset('labels', data=all_labels, dtype=np.int32)
            f.attrs['mean'] = mean
            f.attrs['std'] = std
            f.attrs['num_correct'] = len(correct_hidden)
            f.attrs['num_false'] = len(false_hidden)

        print(f"\nData preprocessing complete!")
        print(f"  Shape: {all_hidden.shape}")
        print(f"  Mean: {mean:.6f}, Std: {std:.6f}")

        return {
            'num_samples': len(all_hidden),
            'num_correct': len(correct_hidden),
            'num_false': len(false_hidden),
            'mean': mean,
            'std': std
        }
