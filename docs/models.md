# Models

SimScaleAI ships three model architectures for robotic manipulation, each registered in a global model registry. Models follow a unified interface: `forward(batch) → dict` for training and `predict(obs) → actions` for inference.

---

## Model Registry

All models are registered by name and can be created dynamically:

```python
from simscaleai.models import ModelRegistry

# List available models
print(ModelRegistry.list())  # ['bc', 'vla']

# Create a model by name
model = ModelRegistry.create("bc", state_dim=20, action_dim=4)

# Create with custom kwargs
model = ModelRegistry.create("vla",
    image_size=64,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    action_dim=4,
)
```

### Registering Custom Models

Use the `@register_model` decorator:

```python
from torch import nn
from simscaleai.models.registry import register_model

@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self, state_dim=20, action_dim=4, **kwargs):
        super().__init__()
        self.net = nn.Linear(state_dim, action_dim)

    def forward(self, batch):
        actions = self.net(batch["observations"]["state"])
        loss = nn.functional.mse_loss(actions, batch["actions"])
        return {"predicted_actions": actions, "loss": loss}

    def predict(self, obs):
        return self.net(obs["state"])
```

---

## Behavior Cloning (BC)

**Supervised imitation learning.** Given (observation, action) pairs from expert demonstrations, learns a policy $\pi(a \mid o)$ via regression.

### Architecture

```
State (obs_dim) ──→ MLP Encoder ──┐
                                  ├──→ Fusion MLP ──→ Action (action_dim)
Image (C,H,W) ──→ CNN Encoder ───┘
                    (optional)
```

- **State encoder**: Multi-layer MLP with GELU activations and dropout
- **Image encoder**: 3-layer CNN (Conv2d 8→4→3 channels, stride 4→2→1) + AdaptiveAvgPool2d(4) + Linear projection
- **Fusion**: Concatenation → MLP → action output

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state_dim` | `int` | `20` | Dimension of state observation vector |
| `action_dim` | `int` | `4` | Dimension of action output |
| `hidden_dim` | `int` | `256` | MLP hidden layer width |
| `n_layers` | `int` | `3` | Number of MLP layers |
| `use_image` | `bool` | `False` | Enable image conditioning |
| `image_channels` | `int` | `3` | Input image channels |
| `image_size` | `int` | `128` | Input image resolution |
| `image_embed_dim` | `int` | `128` | CNN output embedding dimension |
| `dropout` | `float` | `0.1` | Dropout rate |
| `loss_type` | `str` | `"mse"` | Loss function: `"mse"` or `"l1"` |

### Usage

```python
model = ModelRegistry.create("bc",
    state_dim=20,
    action_dim=4,
    use_image=True,
    image_size=128,
)

# Training forward pass
batch = {
    "observations": {
        "state": torch.randn(8, 20),
        "image": torch.randn(8, 3, 128, 128),
    },
    "actions": torch.randn(8, 4),
}

output = model(batch)
# output = {
#     "predicted_actions": Tensor (8, 4),
#     "loss": scalar Tensor,
#     "action_mse": scalar Tensor,
#     "action_mae": scalar Tensor,
# }

# Inference
obs = {"state": torch.randn(1, 20)}
actions = model.predict(obs)  # (1, 4)
```

### Loss Function

$$\mathcal{L}_{\text{BC}} = \begin{cases} \frac{1}{N}\sum_{i=1}^{N} \|a_i - \hat{a}_i\|^2 & \text{if loss\_type = "mse"} \\ \frac{1}{N}\sum_{i=1}^{N} |a_i - \hat{a}_i| & \text{if loss\_type = "l1"} \end{cases}$$

---

## Vision-Language-Action (VLA)

**Frontier robotic foundation model.** Inspired by RT-2 and OpenVLA, this transformer-based model takes image observations and optional language instructions to produce robot actions.

### Architecture

```
Image (C,H,W) ──→ PatchEmbedding ──→ VisionEncoder (ViT) ──→ Visual Tokens
                                                                    │
Language (str) ──→ CharTokenizer ──→ LanguageEncoder ──→ Lang Tokens │
                   (optional)                                        │
State (dim) ──→ Linear Projection ──→ State Token                    │
               (optional)                                            │
                                ┌────────────────────────────────────┘
                                ▼
                     Fusion Transformer
                           ▼
                       CLS Token
                           ▼
                     Action Head (MLP or Diffusion)
                           ▼
                    Actions (action_dim) or (horizon, action_dim)
```

### Components

#### PatchEmbedding

Splits image into non-overlapping patches and projects to embedding dimension with learned positional embeddings.

- Input: `(B, C, H, W)` → Output: `(B, num_patches, embed_dim)`
- `num_patches = (image_size / patch_size)²`

#### VisionEncoder

ViT-style transformer encoder with:

- Pre-norm (`norm_first=True`) transformer layers
- GELU activation in feedforward blocks
- Learnable CLS token prepended to patch sequence
- Output includes CLS + all patch tokens: `(B, num_patches + 1, embed_dim)`

#### SimpleLanguageEncoder

Character-level language encoder for task instructions:

- **Vocabulary**: `" abcdefghijklmnopqrstuvwxyz0123456789.,!?'-"` (40 chars + padding)
- Embedding + positional encoding + transformer encoder
- Output: `(B, seq_len, embed_dim)`

#### Fusion Transformer

Cross-modal transformer that attends over concatenated `[visual_tokens; lang_tokens; state_token]`:

- `num_layers // 2` layers (minimum 2)
- CLS token from vision encoder is used as query summary

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_size` | `int` | `128` | Input image resolution |
| `patch_size` | `int` | `16` | ViT patch size |
| `in_channels` | `int` | `3` | Image channels |
| `embed_dim` | `int` | `256` | Transformer embedding dimension |
| `num_heads` | `int` | `4` | Attention heads |
| `num_layers` | `int` | `4` | Vision encoder layers |
| `mlp_ratio` | `float` | `4.0` | FFN hidden dim multiplier |
| `dropout` | `float` | `0.1` | Dropout rate |
| `use_language` | `bool` | `True` | Enable language conditioning |
| `vocab_size` | `int` | `1000` | Max vocabulary size |
| `max_text_len` | `int` | `64` | Max token sequence length |
| `lang_num_layers` | `int` | `2` | Language encoder layers |
| `action_dim` | `int` | `4` | Output action dimension |
| `action_head_type` | `str` | `"mlp"` | Action head: `"mlp"` or `"diffusion"` |
| `action_horizon` | `int` | `1` | Action chunking horizon |
| `state_dim` | `int` | `0` | State vector dim (0 = disabled) |

### Usage

```python
model = ModelRegistry.create("vla",
    image_size=128,
    embed_dim=256,
    num_heads=4,
    num_layers=4,
    action_dim=4,
    use_language=True,
    action_horizon=1,
)

# Training
batch = {
    "observations": {
        "image": torch.randn(4, 3, 128, 128),
        "state": torch.randn(4, 20),        # optional
    },
    "actions": torch.randn(4, 4),
    "language": ["pick up the red cube"] * 4,  # optional
}

output = model(batch)
# output = {"predicted_actions", "loss", "action_mse"}

# Inference
obs = {"image": torch.randn(1, 3, 128, 128)}
actions = model.predict(obs, language="pick up the red cube")
```

### Action Chunking

When `action_horizon > 1`, the model predicts multiple future actions:

```python
model = ModelRegistry.create("vla", action_horizon=4, action_dim=4)

# Output shape: (batch, horizon, action_dim) = (B, 4, 4)
output = model(batch)
print(output["predicted_actions"].shape)  # (B, 4, 4)
```

### Weight Initialization

VLA uses ViT-style initialization:

- Linear layers: `trunc_normal_(std=0.02)`
- LayerNorm: `weight=1.0, bias=0.0`

### Model Scaling Guide

| Config | embed_dim | heads | layers | Params | Use case |
|--------|-----------|-------|--------|--------|----------|
| Debug | 64 | 2 | 2 | ~200K | Local testing, CI |
| Small | 128 | 4 | 4 | ~2M | Prototyping |
| Medium | 256 | 8 | 8 | ~15M | Single GPU training |
| Large | 512 | 16 | 12 | ~80M | Multi-GPU training |
| XL | 1024 | 16 | 24 | ~350M | Cloud GPU cluster |

---

## Policy Heads

### MLP Action Head

Standard multi-layer perceptron for deterministic action prediction.

```python
from simscaleai.models.policy_heads.mlp_head import MLPActionHead

head = MLPActionHead(
    input_dim=256,      # Feature dimension from backbone
    action_dim=4,       # Output actions
    hidden_dim=256,     # Hidden layer width
    n_layers=2,         # Number of hidden layers
    dropout=0.1,        # Dropout rate
    tanh_output=True,   # Clamp output to [-1, 1]
)

features = torch.randn(8, 256)
actions = head(features)  # (8, 4), values in [-1, 1]
```

### Diffusion Action Head

Implements **Diffusion Policy** (Chi et al., 2023) — uses denoising diffusion for multi-modal action distributions. This is critical for tasks where multiple valid actions exist for the same observation.

**Training:** Forward diffusion adds noise to ground-truth actions at a random timestep; the network learns to predict the noise.

$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]$$

**Inference:** DDPM iterative denoising from pure noise, conditioned on the observation features.

```python
from simscaleai.models.policy_heads.diffusion_head import DiffusionActionHead

head = DiffusionActionHead(
    input_dim=256,
    action_dim=4,
    hidden_dim=256,
    n_layers=4,
    num_diffusion_steps=100,    # Training noise schedule steps
    num_inference_steps=10,     # Denoising steps at inference
    beta_start=1e-4,            # Noise schedule start
    beta_end=0.02,              # Noise schedule end
)

# Training (with targets)
features = torch.randn(8, 256)
targets = torch.randn(8, 4)
output = head(features, target_actions=targets)
loss = output["loss"]

# Inference (without targets)
output = head(features)
actions = output["predicted_actions"]  # (8, 4)
```

**Implementation details:**

- Linear noise schedule: $\beta_t$ linearly spaced from `beta_start` to `beta_end`
- Sinusoidal positional embedding for timestep conditioning
- Residual MLP blocks with skip connections for noise prediction
- Subsampled timesteps during inference for faster generation
- Actions clamped to $[-1, 1]$ after denoising
