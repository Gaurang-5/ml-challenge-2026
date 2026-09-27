"""
PyTorch Neural Entity Matching Architecture.

Deep Neural Network designed for Pairwise Business Entity Matching,
optimized with Weighted Focal Loss to maximize Macro F0.5 (precision-weighted).
Runs with native GPU/MPS acceleration on Apple Silicon and CUDA.
"""

import torch
import torch.nn as nn


class EntityResolutionNet(nn.Module):
    def __init__(self, input_dim: int = 15, hidden_dims: list[int] = [128, 64, 32], dropout: float = 0.2):
        super().__init__()
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.SiLU())  # Swish activation
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting match probability in [0, 1]."""
        logits = self.network(x).squeeze(-1)
        return torch.sigmoid(logits)


class MacroF05Loss(nn.Module):
    """
    Precision-weighted Focal Loss.
    Penalizes false positives (false merges) 2x more than false negatives,
    directly aligning neural gradient descent with the competition F0.5 metric.
    """
    def __init__(self, alpha: float = 0.35, gamma: float = 2.0, fp_weight: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.fp_weight = fp_weight

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
        # Binary focal loss with extra penalty on false positives
        bce_pos = -targets * torch.log(probs) * ((1.0 - probs) ** self.gamma) * self.alpha
        bce_neg = -(1.0 - targets) * torch.log(1.0 - probs) * (probs ** self.gamma) * (1.0 - self.alpha) * self.fp_weight
        loss = bce_pos + bce_neg
        return loss.mean()
