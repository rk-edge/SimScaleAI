"""Diffusion action head — denoising diffusion for action prediction.

Implements Diffusion Policy (Chi et al., 2023) style action generation.
Instead of directly regressing actions, learns to denoise random noise into actions,
which better captures multi-modal action distributions.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionActionHead(nn.Module):
    """Diffusion-based action prediction head.

    During training: adds noise to target actions, predicts the noise.
    During inference: iteratively denoises random noise to produce actions.

    This captures multi-modal action distributions (e.g., "go left OR right")
    which MSE regression averages out to "go straight" (bad!).
    """

    def __init__(
        self,
        input_dim: int = 256,
        action_dim: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 4,
        num_diffusion_steps: int = 100,
        num_inference_steps: int = 10,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_steps = num_diffusion_steps
        self.num_inference_steps = num_inference_steps

        # Noise schedule (linear)
        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Noise prediction network (conditioned on features + timestep)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmbed(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.condition_proj = nn.Linear(input_dim, hidden_dim)

        # Residual MLP for noise prediction
        self.noise_net = nn.ModuleList()
        for _ in range(n_layers):
            self.noise_net.append(
                ResidualBlock(
                    hidden_dim + hidden_dim + action_dim,  # condition + time + noisy_action
                    hidden_dim,
                    action_dim,
                )
            )
        self.final = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        features: torch.Tensor,
        target_actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        During training (target_actions provided):
            - Add noise to target actions
            - Predict the noise
            - Return MSE loss on noise prediction

        During inference (target_actions is None):
            - Denoise from random noise
            - Return predicted actions
        """
        if target_actions is not None:
            return self._training_step(features, target_actions)
        else:
            return {"predicted_actions": self._inference(features)}

    def _training_step(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        B = features.shape[0]
        device = features.device

        # Random timestep for each sample
        t = torch.randint(0, self.num_steps, (B,), device=device)

        # Add noise to actions
        noise = torch.randn_like(actions)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        noisy_actions = sqrt_alpha * actions + sqrt_one_minus * noise

        # Predict noise
        pred_noise = self._predict_noise(features, noisy_actions, t)

        loss = F.mse_loss(pred_noise, noise)
        return {"loss": loss, "predicted_actions": actions - pred_noise}

    @torch.no_grad()
    def _inference(self, features: torch.Tensor) -> torch.Tensor:
        """DDPM sampling: iteratively denoise random noise."""
        B = features.shape[0]
        device = features.device

        # Start from pure noise
        x = torch.randn(B, self.action_dim, device=device)

        # Subsample timesteps for faster inference
        step_size = max(1, self.num_steps // self.num_inference_steps)
        timesteps = list(range(0, self.num_steps, step_size))[::-1]

        for t_val in timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            pred_noise = self._predict_noise(features, x, t)

            alpha = self.alphas[t_val]
            alpha_cumprod = self.alphas_cumprod[t_val]
            beta = self.betas[t_val]

            # DDPM update
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise
            )

            # Add noise (except at last step)
            if t_val > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta) * noise

        return torch.clamp(x, -1.0, 1.0)

    def _predict_noise(
        self, features: torch.Tensor, noisy_actions: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise given condition, noisy actions, and timestep."""
        cond = self.condition_proj(features)
        time_emb = self.time_embed(t)

        h = torch.cat([cond, time_emb, noisy_actions], dim=-1)

        for block in self.noise_net:
            h = block(h)

        return self.final(h)


class SinusoidalPosEmbed(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualBlock(nn.Module):
    """Residual MLP block for noise prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.skip = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)
