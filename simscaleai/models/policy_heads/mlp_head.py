"""MLP action head — simple feedforward action predictor."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPActionHead(nn.Module):
    """Standard MLP head that maps features to action predictions.

    Used as the default action head in BC and VLA models.
    """

    def __init__(
        self,
        input_dim: int = 256,
        action_dim: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        tanh_output: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, action_dim))
        if tanh_output:
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map features to actions. (B, input_dim) → (B, action_dim)"""
        return self.net(features)
