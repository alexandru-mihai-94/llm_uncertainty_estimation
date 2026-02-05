"""
Model Architecture Module for Factoscope

This module contains the neural network architectures for encoding hidden states,
ranks, and probabilities, and combining them into a unified factuality representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HiddenStateEncoder(nn.Module):
    """
    Encodes hidden states using 1D convolutions.

    Processes layer-wise hidden states from the LLM using convolutional layers
    to extract meaningful patterns across layers.
    """

    def __init__(self, num_layers: int, hidden_dim: int, emb_dim: int = 32):
        """
        Initialize the hidden state encoder.

        Args:
            num_layers: Number of layers in the LLM
            hidden_dim: Dimensionality of hidden states
            emb_dim: Output embedding dimension
        """
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch, num_layers, hidden_dim)

        Returns:
            Encoded tensor of shape (batch, emb_dim)
        """
        # x shape: (batch, num_layers, hidden_dim)
        x = x.transpose(1, 2)  # (batch, hidden_dim, num_layers)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (batch, 128)
        x = self.fc(x)
        return x


class RankEncoder(nn.Module):
    """
    Encodes rank information using a simple MLP.

    Processes the rank of the generated token in the model's probability
    distribution to capture confidence-related information.
    """

    def __init__(self, emb_dim: int = 32):
        """
        Initialize the rank encoder.

        Args:
            emb_dim: Output embedding dimension
        """
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch,) or (batch, 1)

        Returns:
            Encoded tensor of shape (batch, emb_dim)
        """
        # x shape: (batch,) or (batch, 1)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ProbEncoder(nn.Module):
    """
    Encodes top-k token probabilities.

    Processes the probabilities of the top-k most likely tokens to capture
    the model's confidence distribution.
    """

    def __init__(self, emb_dim: int = 32):
        """
        Initialize the probability encoder.

        Args:
            emb_dim: Output embedding dimension
        """
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch, 10)

        Returns:
            Encoded tensor of shape (batch, emb_dim)
        """
        # x shape: (batch, 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FactoscopeModel(nn.Module):
    """
    Complete Factoscope model combining all feature encoders.

    This model integrates hidden states, ranks, and probabilities into a
    unified representation for factuality assessment using metric learning.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        emb_dim: int = 32,
        final_dim: int = 64
    ):
        """
        Initialize the Factoscope model.

        Args:
            num_layers: Number of layers in the LLM
            hidden_dim: Dimensionality of hidden states
            emb_dim: Intermediate embedding dimension for each encoder
            final_dim: Final output embedding dimension
        """
        super().__init__()
        self.hidden_encoder = HiddenStateEncoder(num_layers, hidden_dim, emb_dim)
        self.rank_encoder = RankEncoder(emb_dim)
        self.prob_encoder = ProbEncoder(emb_dim)

        # Combine all encodings
        self.combiner = nn.Sequential(
            nn.Linear(emb_dim * 3, final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, final_dim)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        rank: torch.Tensor,
        prob: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            hidden: Hidden states tensor of shape (batch, num_layers, hidden_dim)
            rank: Rank tensor of shape (batch,) or (batch, 1)
            prob: Probability tensor of shape (batch, 10)

        Returns:
            Normalized embedding tensor of shape (batch, final_dim)
        """
        h_emb = self.hidden_encoder(hidden)
        r_emb = self.rank_encoder(rank)
        p_emb = self.prob_encoder(prob)

        combined = torch.cat([h_emb, r_emb, p_emb], dim=1)
        output = self.combiner(combined)

        # Normalize for metric learning
        output = F.normalize(output, p=2, dim=1)
        return output
