"""
Training Module for Factoscope

This module handles training the factuality classifier using triplet loss
and metric learning with a support set for evaluation.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix


class FactoscopeDataset(Dataset):
    """
    Dataset for triplet learning.

    Generates triplets (anchor, positive, negative) on-the-fly where:
    - anchor: A sample
    - positive: A sample with the same label as anchor
    - negative: A sample with a different label than anchor
    """

    def __init__(
        self,
        hidden_states: np.ndarray,
        ranks: np.ndarray,
        probs: np.ndarray,
        labels: np.ndarray
    ):
        """
        Initialize dataset.

        Args:
            hidden_states: Hidden state features of shape (N, layers, hidden_dim)
            ranks: Rank features of shape (N,)
            probs: Probability features of shape (N, 10)
            labels: Labels of shape (N,) - 1 for correct, 0 for false
        """
        self.hidden_states = torch.from_numpy(hidden_states).float()
        self.ranks = torch.from_numpy(ranks).float()
        self.probs = torch.from_numpy(probs).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a triplet: anchor, positive, negative.

        Args:
            idx: Index of the anchor sample

        Returns:
            Tuple of (anchor_hidden, anchor_rank, anchor_prob,
                     positive_hidden, positive_rank, positive_prob,
                     negative_hidden, negative_rank, negative_prob,
                     anchor_label, positive_label, negative_label)
        """
        anchor_label = self.labels[idx]

        # Find positive (same label)
        pos_indices = torch.where(self.labels == anchor_label)[0]
        pos_idx = pos_indices[torch.randint(len(pos_indices), (1,))].item()

        # Find negative (different label)
        neg_indices = torch.where(self.labels != anchor_label)[0]
        neg_idx = neg_indices[torch.randint(len(neg_indices), (1,))].item()

        return (
            self.hidden_states[idx], self.ranks[idx], self.probs[idx],
            self.hidden_states[pos_idx], self.ranks[pos_idx], self.probs[pos_idx],
            self.hidden_states[neg_idx], self.ranks[neg_idx], self.probs[neg_idx],
            self.labels[idx], self.labels[pos_idx], self.labels[neg_idx]
        )


class FactoscopeTrainer:
    """
    Trainer for the Factoscope model.

    Handles training with triplet loss and evaluation using nearest neighbor
    classification in the learned embedding space.
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize the trainer.

        Args:
            model: The Factoscope model to train
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    def train_epoch(self, train_loader, optimizer) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer for updating model parameters

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            # Unpack triplet
            anchor_h, anchor_r, anchor_p, pos_h, pos_r, pos_p, \
                neg_h, neg_r, neg_p, _, _, _ = batch

            # Move to device
            anchor_h = anchor_h.to(self.device)
            anchor_r = anchor_r.to(self.device)
            anchor_p = anchor_p.to(self.device)
            pos_h = pos_h.to(self.device)
            pos_r = pos_r.to(self.device)
            pos_p = pos_p.to(self.device)
            neg_h = neg_h.to(self.device)
            neg_r = neg_r.to(self.device)
            neg_p = neg_p.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            anchor_emb = self.model(anchor_h, anchor_r, anchor_p)
            pos_emb = self.model(pos_h, pos_r, pos_p)
            neg_emb = self.model(neg_h, neg_r, neg_p)

            # Compute loss
            loss = self.criterion(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, test_loader, support_loader) -> Dict:
        """
        Evaluate model using nearest neighbor in embedding space.

        Uses a support set to perform classification by finding the nearest
        neighbor in the embedding space and using its label.

        Args:
            test_loader: DataLoader for test data
            support_loader: DataLoader for support set

        Returns:
            Dictionary with metrics:
                - accuracy: Overall accuracy
                - precision: Precision for correct class
                - recall: Recall for correct class
                - f1: F1 score
                - TP, FP, TN, FN: Confusion matrix values
        """
        self.model.eval()

        # Build support set embeddings
        support_embeddings = []
        support_labels = []

        with torch.no_grad():
            for batch in support_loader:
                h, r, p, _, _, _, _, _, _, label, _, _ = batch
                h = h.to(self.device)
                r = r.to(self.device)
                p = p.to(self.device)

                emb = self.model(h, r, p)
                support_embeddings.append(emb)
                support_labels.append(label)

        support_embeddings = torch.cat(support_embeddings, dim=0)
        support_labels = torch.cat(support_labels, dim=0)

        # Evaluate on test set
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in test_loader:
                h, r, p, _, _, _, _, _, _, label, _, _ = batch
                h = h.to(self.device)
                r = r.to(self.device)
                p = p.to(self.device)

                emb = self.model(h, r, p)

                # Find nearest neighbor in support set
                distances = torch.cdist(emb, support_embeddings, p=2)
                nearest_idx = torch.argmin(distances, dim=1)
                pred = support_labels[nearest_idx]

                y_true.extend(label.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

        # Calculate metrics
        conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        TN, FP, FN, TP = conf_matrix.ravel()

        accuracy = (TP + TN) / (TP + FP + FN + TN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN
        }
