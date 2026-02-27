"""Vision-Language-Action (VLA) model — frontier robotic foundation model.

Inspired by RT-2 and OpenVLA: takes an image + language instruction as input,
outputs robot actions. Uses a pretrained vision encoder (CLIP/SigLIP-style)
fused with a transformer for language understanding and action prediction.

Architecture:
    image → Vision Encoder (ViT) → visual tokens
    language → Tokenizer → Token Embeddings → language tokens
    [visual tokens; language tokens] → Transformer Decoder → Action Head → actions
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from simscaleai.models.registry import register_model


class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dim.

    This is the first step of a Vision Transformer (ViT).
    """

    def __init__(self, image_size: int = 128, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        return x


class VisionEncoder(nn.Module):
    """Lightweight Vision Transformer encoder.

    For debug/local: small ViT (2-4 layers, 128-256 dim).
    For production: swap to pretrained CLIP/SigLIP via HuggingFace.
    """

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to visual token sequence.

        Args:
            images: (B, C, H, W)

        Returns:
            visual_tokens: (B, num_patches + 1, embed_dim)
        """
        patches = self.patch_embed(images)
        B = patches.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = self.encoder(tokens)
        return self.norm(tokens)


class SimpleLanguageEncoder(nn.Module):
    """Lightweight language encoder using learned embeddings + transformer.

    For debug/local: small vocab + 2-layer transformer.
    For production: swap to pretrained LLM tokenizer + backbone.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        max_len: int = 64,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        self.max_len = max_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Simple character-level tokenizer for demo purposes
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build a simple character-level vocabulary."""
        chars = " abcdefghijklmnopqrstuvwxyz0123456789.,!?'-"
        self._char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 = padding
        self._pad_idx = 0

    def tokenize(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Convert text strings to token indices."""
        batch_tokens = []
        for text in texts:
            tokens = [self._char_to_idx.get(c, 0) for c in text.lower()[:self.max_len]]
            # Pad to max_len
            tokens = tokens + [self._pad_idx] * (self.max_len - len(tokens))
            batch_tokens.append(tokens)
        return torch.tensor(batch_tokens, dtype=torch.long, device=device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode token IDs to language features.

        Args:
            token_ids: (B, seq_len) long tensor

        Returns:
            language_tokens: (B, seq_len, embed_dim)
        """
        seq_len = token_ids.shape[1]
        x = self.token_embed(token_ids) + self.pos_embed[:, :seq_len]
        x = self.encoder(x)
        return self.norm(x)


@register_model("vla")
class VisionLanguageAction(nn.Module):
    """Vision-Language-Action (VLA) model.

    Fuses visual perception and language understanding to produce
    robot actions. Core architecture for robotic foundation models.

    Supports:
    - Image-only mode (no language)
    - Image + language mode (full VLA)
    - Configurable action heads (MLP, diffusion)
    - Debug (tiny) and full-scale configurations
    """

    def __init__(
        self,
        # Vision
        image_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        # Architecture
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        # Language
        use_language: bool = True,
        vocab_size: int = 1000,
        max_text_len: int = 64,
        lang_num_layers: int = 2,
        # Action
        action_dim: int = 4,
        action_head_type: str = "mlp",  # 'mlp' or 'diffusion'
        action_horizon: int = 1,  # Number of future actions to predict (chunking)
        # State
        state_dim: int = 0,  # Additional state input (joint pos, etc.)
    ):
        super().__init__()
        self.use_language = use_language
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Language encoder
        if use_language:
            self.language_encoder = SimpleLanguageEncoder(
                vocab_size=vocab_size,
                max_len=max_text_len,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=lang_num_layers,
                dropout=dropout,
            )

        # Optional state projection
        self.state_proj = None
        if state_dim > 0:
            self.state_proj = nn.Sequential(
                nn.Linear(state_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

        # Cross-modal fusion transformer
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(
            fusion_layer, num_layers=max(num_layers // 2, 2)
        )

        # Action head
        total_action_dim = action_dim * action_horizon
        if action_head_type == "mlp":
            self.action_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, total_action_dim),
                nn.Tanh(),  # Actions in [-1, 1]
            )
        else:
            # Placeholder for diffusion head
            self.action_head = nn.Sequential(
                nn.Linear(embed_dim, total_action_dim),
                nn.Tanh(),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with truncated normal (ViT-style)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch: Dict with:
                - observations: {'image': (B,C,H,W), 'state': (B,D) optional}
                - actions: (B, action_dim) target actions (for training)
                - language: list[str] or token_ids (B, seq_len) (optional)

        Returns:
            Dict with 'loss', 'predicted_actions', and metrics
        """
        obs = batch["observations"]
        device = next(self.parameters()).device

        # Encode vision
        image = obs["image"]  # (B, C, H, W)
        visual_tokens = self.vision_encoder(image)  # (B, N_v, D)

        # Build token sequence for fusion
        all_tokens = [visual_tokens]

        # Encode language if provided
        if self.use_language and "language" in batch:
            lang_input = batch["language"]
            if isinstance(lang_input, list):
                # Tokenize strings
                token_ids = self.language_encoder.tokenize(lang_input, device)
            else:
                token_ids = lang_input
            lang_tokens = self.language_encoder(token_ids)  # (B, N_l, D)
            all_tokens.append(lang_tokens)

        # Project state if available
        if self.state_proj is not None and "state" in obs:
            state = obs["state"]  # (B, state_dim)
            state_token = self.state_proj(state).unsqueeze(1)  # (B, 1, D)
            all_tokens.append(state_token)

        # Concatenate all tokens and fuse
        fused = torch.cat(all_tokens, dim=1)  # (B, N_total, D)
        fused = self.fusion_transformer(fused)  # (B, N_total, D)

        # Use CLS token (first token from vision) for action prediction
        cls_feat = fused[:, 0]  # (B, D)

        # Predict actions
        pred_actions = self.action_head(cls_feat)  # (B, action_dim * horizon)

        # Reshape for action chunking
        if self.action_horizon > 1:
            pred_actions = pred_actions.view(-1, self.action_horizon, self.action_dim)

        result: dict[str, torch.Tensor] = {"predicted_actions": pred_actions}

        # Compute loss
        if "actions" in batch:
            target = batch["actions"]
            loss = F.mse_loss(pred_actions, target)
            result["loss"] = loss

            with torch.no_grad():
                result["action_mse"] = F.mse_loss(pred_actions, target)

        return result

    def predict(
        self,
        obs: dict[str, torch.Tensor],
        language: str | None = None,
    ) -> torch.Tensor:
        """Predict actions for deployment.

        Args:
            obs: Observation dict with 'image' and optionally 'state'
            language: Natural language instruction

        Returns:
            Predicted actions (action_dim,) or (horizon, action_dim)
        """
        self.eval()
        with torch.no_grad():
            batch: dict[str, Any] = {"observations": obs}
            if language:
                batch["language"] = [language]
            result = self.forward(batch)
        actions = result["predicted_actions"]
        if self.action_horizon == 1:
            return actions.squeeze(0)
        return actions.squeeze(0)  # (horizon, action_dim)
