# API Reference

Complete reference for all public classes, methods, and functions in SimScaleAI.

---

## `simscaleai.sim` — Simulation

### `simscaleai.sim.base_env`

#### `CameraConfig`

```python
@dataclass
class CameraConfig:
    name: str = "wrist_cam"
    width: int = 128
    height: int = 128
    fov: float = 60.0
    render_depth: bool = True
    render_segmentation: bool = False
```

#### `SimConfig`

```python
@dataclass
class SimConfig:
    xml_path: str = ""
    dt: float = 0.002
    control_dt: float = 0.05
    max_episode_steps: int = 200
    cameras: list[CameraConfig] = field(default_factory=lambda: [CameraConfig()])
    domain_randomization: bool = False
    seed: int | None = None
```

#### `BaseRobotEnv`

```python
class BaseRobotEnv(gymnasium.Env, abc.ABC):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config: SimConfig, render_mode: str | None = None) -> None
    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None
             ) -> tuple[dict[str, np.ndarray], dict[str, Any]]
    def step(self, action: np.ndarray
            ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]
    def render_camera(self, camera_name: str | None = None,
                      depth: bool = False, segmentation: bool = False
                     ) -> dict[str, np.ndarray]
    def render(self) -> np.ndarray | None
    def close(self) -> None

    # Override in subclasses
    @abc.abstractmethod
    def _build_observation_space(self) -> spaces.Space: ...
    @abc.abstractmethod
    def _build_action_space(self) -> spaces.Space: ...
    @abc.abstractmethod
    def _get_obs(self) -> dict[str, np.ndarray]: ...
    @abc.abstractmethod
    def _get_reward(self) -> float: ...
    @abc.abstractmethod
    def _is_terminated(self) -> bool: ...
    @abc.abstractmethod
    def _reset_task(self, rng: np.random.Generator) -> None: ...

    # Optional overrides
    def _apply_action(self, action: np.ndarray) -> None: ...
    def _get_info(self) -> dict[str, Any]: ...
    def _apply_domain_randomization(self, rng: np.random.Generator) -> None: ...
```

### `simscaleai.sim.factory`

```python
def make_env(env_name: str, render_mode: str | None = None, **kwargs) -> gymnasium.Env
def register_env(name: str, cls: type) -> None
def list_envs() -> list[str]
```

### `simscaleai.sim.envs.reach_env`

```python
class ReachEnv(BaseRobotEnv):
    def __init__(self, config: SimConfig | None = None,
                 render_mode: str | None = None) -> None
```

**Observation space:** `Dict(joint_pos=Box(7,), joint_vel=Box(7,), ee_pos=Box(3,), target_pos=Box(3,), image=Box(H,W,3)?)`

**Action space:** `Box(-1, 1, shape=(4,))` — 3D EE delta + gripper

**Info:** `{"step": int, "distance": float, "success": bool}`

### `simscaleai.sim.envs.pick_place_env`

```python
class PickPlaceEnv(BaseRobotEnv):
    def __init__(self, config: SimConfig | None = None,
                 render_mode: str | None = None) -> None
```

**Observation space:** `Dict(joint_pos=Box(7,), joint_vel=Box(7,), ee_pos=Box(3,), object_pos=Box(3,), object_quat=Box(4,), target_pos=Box(3,), gripper_state=Box(1,), image=Box(H,W,3)?)`

**Action space:** `Box(-1, 1, shape=(4,))` — 3D EE delta + gripper

**Info:** `{"step": int, "reach_dist": float, "place_dist": float, "grasped": bool, "lifted": bool, "success": bool}`

---

## `simscaleai.models` — Model Architectures

### `simscaleai.models.registry`

```python
def register_model(name: str) -> Callable    # Decorator
def create_model(name: str, **kwargs) -> nn.Module
def list_models() -> list[str]

class ModelRegistry:
    register = staticmethod(register_model)
    create = staticmethod(create_model)
    list = staticmethod(list_models)
```

### `simscaleai.models.bc`

```python
@register_model("bc")
class BehaviorCloning(nn.Module):
    def __init__(self, state_dim: int = 20, action_dim: int = 4,
                 hidden_dim: int = 256, n_layers: int = 3,
                 use_image: bool = False, image_channels: int = 3,
                 image_size: int = 128, image_embed_dim: int = 128,
                 dropout: float = 0.1, loss_type: str = "mse") -> None

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]
    # Returns: {"predicted_actions", "loss", "action_mse", "action_mae"}

    def predict(self, obs: dict[str, torch.Tensor]) -> torch.Tensor
```

### `simscaleai.models.vla`

```python
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=128, patch_size=16,
                 in_channels=3, embed_dim=256) -> None
    def forward(self, x: torch.Tensor) -> torch.Tensor
    # (B, C, H, W) → (B, num_patches, embed_dim)

class VisionEncoder(nn.Module):
    def __init__(self, image_size=128, patch_size=16, in_channels=3,
                 embed_dim=256, num_heads=4, num_layers=4,
                 mlp_ratio=4.0, dropout=0.1) -> None
    def forward(self, images: torch.Tensor) -> torch.Tensor
    # (B, C, H, W) → (B, num_patches+1, embed_dim)

class SimpleLanguageEncoder(nn.Module):
    def __init__(self, vocab_size=1000, max_len=64, embed_dim=256,
                 num_heads=4, num_layers=2, dropout=0.1) -> None
    def tokenize(self, texts: list[str], device: torch.device) -> torch.Tensor
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor

@register_model("vla")
class VisionLanguageAction(nn.Module):
    def __init__(self, image_size=128, patch_size=16, in_channels=3,
                 embed_dim=256, num_heads=4, num_layers=4, mlp_ratio=4.0,
                 dropout=0.1, use_language=True, vocab_size=1000,
                 max_text_len=64, lang_num_layers=2, action_dim=4,
                 action_head_type="mlp", action_horizon=1,
                 state_dim=0) -> None

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]
    # Returns: {"predicted_actions", "loss", "action_mse"}

    def predict(self, obs: dict[str, torch.Tensor],
                language: str | None = None) -> torch.Tensor
```

### `simscaleai.models.policy_heads.mlp_head`

```python
class MLPActionHead(nn.Module):
    def __init__(self, input_dim=256, action_dim=4, hidden_dim=256,
                 n_layers=2, dropout=0.1, tanh_output=True) -> None
    def forward(self, features: torch.Tensor) -> torch.Tensor
    # (B, input_dim) → (B, action_dim)
```

### `simscaleai.models.policy_heads.diffusion_head`

```python
class SinusoidalPosEmbed(nn.Module):
    def __init__(self, dim: int) -> None
    def forward(self, t: torch.Tensor) -> torch.Tensor

class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None
    def forward(self, x: torch.Tensor) -> torch.Tensor

class DiffusionActionHead(nn.Module):
    def __init__(self, input_dim=256, action_dim=4, hidden_dim=256,
                 n_layers=4, num_diffusion_steps=100,
                 num_inference_steps=10, beta_start=1e-4,
                 beta_end=0.02) -> None

    def forward(self, features: torch.Tensor,
                target_actions: torch.Tensor | None = None
               ) -> dict[str, torch.Tensor]
    # Returns: {"loss", "predicted_actions"}
```

---

## `simscaleai.training` — Training Infrastructure

### `simscaleai.training.trainer`

```python
@dataclass
class TrainConfig:
    model_name: str = "bc"
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    dataset_path: str = ""
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    max_steps: int = 100_000
    grad_clip: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    distributed: bool = False
    backend: str = "nccl"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5000
    resume_from: str | None = None
    log_every: int = 100
    eval_every: int = 1000
    use_wandb: bool = False
    wandb_project: str = "simscaleai"
    wandb_run_name: str | None = None
    device: str = "auto"

    @property
    def resolved_device(self) -> str: ...
    @property
    def amp_torch_dtype(self) -> torch.dtype: ...

class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 config: TrainConfig, eval_loader: DataLoader | None = None,
                 eval_fn: Callable | None = None) -> None
    def train(self) -> dict[str, float]
    def evaluate(self) -> dict[str, float]
    def save_checkpoint(self, step: int, is_final: bool = False) -> Path
    def load_checkpoint(self, path: str) -> None
```

### `simscaleai.training.data.dataset`

```python
class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str | Path, seq_len: int = 1,
                 obs_keys: list[str] | None = None,
                 include_language: bool = False,
                 transform: Any = None) -> None
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]
    # Returns: {"observations": {...}, "actions": Tensor, "language"?: str}

class DummyTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int = 1000, obs_dim: int = 20,
                 action_dim: int = 4, image_size: tuple[int, int] = (128, 128),
                 include_image: bool = False, include_language: bool = False,
                 seed: int = 42) -> None
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]
```

---

## `simscaleai.rl` — Reinforcement Learning

### `simscaleai.rl.agents.ppo`

```python
@dataclass
class PPOConfig:
    hidden_dim: int = 256
    n_layers: int = 2
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    n_steps: int = 2048
    n_epochs: int = 10
    batch_size: int = 64
    total_timesteps: int = 1_000_000
    log_every: int = 10

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dim: int = 256, n_layers: int = 2) -> None
    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]
    def get_action(self, obs: torch.Tensor, deterministic: bool = False
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]

class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int, action_dim: int,
                 device: str) -> None
    def add(self, obs, action, reward, done, log_prob, value) -> None
    def compute_gae(self, last_value, gamma: float, gae_lambda: float) -> None
    def get_batches(self, batch_size: int) -> Generator
    def reset(self) -> None

class PPOAgent:
    def __init__(self, obs_dim: int, action_dim: int,
                 config: PPOConfig | None = None,
                 device: str = "cpu") -> None
    def train(self, env) -> dict[str, list[float]]
    def predict(self, obs, deterministic: bool = True) -> np.ndarray
    def save(self, path: str) -> None
    def load(self, path: str) -> None
```

### `simscaleai.rl.evaluator`

```python
@dataclass
class EvalConfig:
    n_episodes: int = 20
    max_steps: int = 200
    deterministic: bool = True
    render: bool = False
    save_videos: bool = False
    video_dir: str = "eval_videos"

def evaluate_policy(env, predict_fn: Callable,
                    config: EvalConfig | None = None) -> dict[str, float]
# Returns: {"success_rate", "mean_reward", "std_reward",
#           "mean_length", "min_reward", "max_reward"}
```

### `simscaleai.rl.rewards.rewards`

```python
class RewardFunction(abc.ABC):
    @abc.abstractmethod
    def compute(self, obs: dict, action: np.ndarray, info: dict) -> float: ...

class DistanceReward(RewardFunction):
    def __init__(self, key_a="ee_pos", key_b="target_pos",
                 scale: float = 1.0) -> None

class SuccessBonus(RewardFunction):
    def __init__(self, key_a="ee_pos", key_b="target_pos",
                 threshold: float = 0.05, bonus: float = 1.0) -> None

class ActionPenalty(RewardFunction):
    def __init__(self, scale: float = 0.01) -> None

class CompositeReward(RewardFunction):
    def __init__(self, rewards: list[tuple[RewardFunction, float]]) -> None
```

---

## `simscaleai.datagen` — Data Generation

### `simscaleai.datagen.generator`

```python
def generate_dataset(
    env_name: str = "reach",
    n_episodes: int = 100,
    output_path: str = "data/dataset.h5",
    policy_type: str = "random",
    domain_randomization: bool = True,
    max_steps: int = 200,
    seed: int = 42,
) -> dict[str, Any]
# Returns: {"total_episodes", "total_steps", "mean_episode_length",
#           "mean_reward", "success_rate", "output_path", "file_size_mb"}
```
