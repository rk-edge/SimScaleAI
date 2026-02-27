"""Behavior Cloning (BC) — baseline imitation learning model.

Given (observation, action) pairs from expert demonstrations,
learns a policy π(a|o) via supervised regression.

This is the simplest foundation model — every robotics project starts here.
It validates that the data pipeline, training loop, and env interface all work.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from simscaleai.models.registry import register_model


@register_model("bc")
class BehaviorCloning(nn.Module):
    """Behavior Cloning with optional image encoder.

    Supports:
    - State-only BC (joint positions, ee position, etc.)
    - Image-conditioned BC (CNN encoder + state)
    - Mixed precision compatible

    Architecture:
        state → MLP ──┐
                       ├── fusion MLP → action
        image → CNN ──┘
    """

    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 3,
        use_image: bool = False,
        image_channels: int = 3,
        image_size: int = 128,
        image_embed_dim: int = 128,
        dropout: float = 0.1,
        loss_type: str = "mse",  # 'mse' or 'l1'
    ):
        super().__init__()
        self.use_image = use_image
        self.loss_type = loss_type

        # State encoder
        self.state_encoder = self._build_mlp(
            state_dim, hidden_dim, hidden_dim, n_layers=2, dropout=dropout
        )

        # Image encoder (lightweight CNN)
        if use_image:
            self.image_encoder = nn.Sequential(
                nn.Conv2d(image_channels, 32, 8, stride=4, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, image_embed_dim),
                nn.ReLU(),
            )
            fusion_input_dim = hidden_dim + image_embed_dim
        else:
            fusion_input_dim = hidden_dim

        # Action prediction head
        self.action_head = self._build_mlp(
            fusion_input_dim, hidden_dim, action_dim, n_layers=n_layers, dropout=dropout
        )

        # Loss function
        self.loss_fn = nn.MSELoss() if loss_type == "mse" else nn.L1Loss()

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch: Dict with 'observations' (dict of tensors) and 'actions' (B, action_dim)

        Returns:
            Dict with 'loss', 'predicted_actions', and optional metrics
        """
        obs = batch["observations"]

        # Encode state — either a single 'state' key or concat all non-image keys
        if "state" in obs:
            state = obs["state"]  # (B, state_dim)
        else:
            # Auto-concatenate all non-image observation keys
            parts = []
            for k in sorted(obs.keys()):
                if k == "image":
                    continue
                v = obs[k]
                if v.dim() == 1:
                    v = v.unsqueeze(0)
                parts.append(v.float())
            state = torch.cat(parts, dim=-1)  # (B, total_state_dim)
        state_feat = self.state_encoder(state)

        # Encode image if available
        if self.use_image and "image" in obs:
            image = obs["image"]  # (B, C, H, W)
            image_feat = self.image_encoder(image)
            feat = torch.cat([state_feat, image_feat], dim=-1)
        else:
            feat = state_feat

        # Predict actions
        pred_actions = self.action_head(feat)

        result: dict[str, torch.Tensor] = {"predicted_actions": pred_actions}

        # Compute loss if target actions provided
        if "actions" in batch:
            target = batch["actions"]
            loss = self.loss_fn(pred_actions, target)
            result["loss"] = loss

            # Extra metrics
            with torch.no_grad():
                result["action_mse"] = nn.functional.mse_loss(pred_actions, target)
                result["action_mae"] = nn.functional.l1_loss(pred_actions, target)

        return result

    def predict(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict actions for inference (no loss computation)."""
        self.eval()
        with torch.no_grad():
            result = self.forward({"observations": obs})
        return result["predicted_actions"]

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
