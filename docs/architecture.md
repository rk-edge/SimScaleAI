# Architecture

Design patterns, system internals, and extension guide for SimScaleAI.

---

## Design Principles

1. **Config-driven** — Every subsystem uses `@dataclass` configs, enabling YAML/CLI/programmatic configuration without code changes
2. **Registry pattern** — Models and environments are registered by name, enabling dynamic instantiation and discovery
3. **Template method** — `BaseRobotEnv` defines the simulation loop; subclasses implement only task-specific logic
4. **Composability** — Reward functions, policy heads, and training components are modular and can be mixed freely
5. **Progressive complexity** — Debug configs run on any laptop; full configs scale to multi-GPU clusters

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                            CLI Layer                                │
│                    simscaleai/tools/cli.py                          │
│               Typer commands → Rich formatted output                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐             │
│  │  Simulation   │  │   Training   │  │    RL Agent    │             │
│  │              │  │              │  │               │             │
│  │ BaseRobotEnv │  │   Trainer    │  │  PPOAgent     │             │
│  │ ↳ ReachEnv   │  │  TrainConfig │  │  PPOConfig    │             │
│  │ ↳ PickPlace  │  │  AMP/DDP     │  │  ActorCritic  │             │
│  │              │  │              │  │  RolloutBuffer│             │
│  │ SimConfig    │  │  TrajectoryDS │  │               │             │
│  │ CameraConfig │  │  DummyDS     │  │  CompositeRwd │             │
│  │              │  │              │  │  Evaluator    │             │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘             │
│         │                 │                   │                     │
│  ┌──────┴───────┐  ┌──────┴───────────────────┴──────┐             │
│  │   MuJoCo     │  │        Model Registry            │             │
│  │   Physics    │  │                                  │             │
│  │   Backend    │  │  @register_model("bc")  → BC     │             │
│  │              │  │  @register_model("vla") → VLA    │             │
│  │  MJCF XML    │  │                                  │             │
│  │  Renderer    │  │  Policy Heads:                   │             │
│  │              │  │    MLPActionHead                 │             │
│  │              │  │    DiffusionActionHead           │             │
│  └──────────────┘  └────────────────────────────────  ┘             │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                        Data Pipeline                                │
│   DataGen (HDF5 export) → TrajectoryDataset → DataLoader → Trainer  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Patterns

### 1. Registry Pattern

Both models and environments use module-level dictionaries with lazy registration:

```python
# simscaleai/models/registry.py
_MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}

def register_model(name: str):
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def create_model(name: str, **kwargs) -> nn.Module:
    _ensure_registered()  # Lazy import on first use
    return _MODEL_REGISTRY[name](**kwargs)
```

**Why lazy registration?** Avoids circular imports and import-time side effects. Models are only imported when first needed.

### 2. Template Method (BaseRobotEnv)

The base environment defines the complete simulation loop — subclasses only implement 6 abstract methods:

```python
class BaseRobotEnv(gym.Env, ABC):
    def reset(self, ...):           # Template: reset state → DR → _reset_task → physics → _get_obs
        ...
    def step(self, action):         # Template: _apply_action → physics × N → _get_obs → _get_reward
        ...

    # Subclass hooks:
    @abstractmethod
    def _build_observation_space(self): ...  # Define obs space
    @abstractmethod
    def _build_action_space(self): ...      # Define action space
    @abstractmethod
    def _get_obs(self): ...                 # Read current observation
    @abstractmethod
    def _get_reward(self): ...              # Compute reward
    @abstractmethod
    def _is_terminated(self): ...           # Check success/failure
    @abstractmethod
    def _reset_task(self, rng): ...         # Randomize task parameters
```

### 3. Composable Rewards

Reward functions follow the Strategy pattern — combine simple building blocks into complex reward signals:

```python
reward = CompositeReward([
    (DistanceReward("ee_pos", "target_pos"), 1.0),     # Reach
    (SuccessBonus("ee_pos", "target_pos", 0.05), 5.0), # Bonus
    (ActionPenalty(0.01), 0.5),                         # Smooth
])
```

### 4. Unified Model Interface

All models implement the same `forward` / `predict` contract:

```python
# Training (with loss computation)
output = model(batch)
# output["loss"]               → scalar for backward()
# output["predicted_actions"]  → (B, action_dim) or (B, horizon, action_dim)

# Inference (no loss, no grad)
actions = model.predict(obs)
```

This allows the Trainer to be completely model-agnostic — it just calls `model(batch)` and reads `output["loss"]`.

### 5. Jacobian-Transpose IK

The reach environment maps Cartesian end-effector deltas to joint-space commands using the Jacobian transpose:

```python
def _apply_action(self, action):
    delta_pos = action[:3] * 0.05  # Scale to meters

    # Compute Jacobian at EE site
    jacp = np.zeros((3, self.model.nv))
    mujoco.mj_jacSite(self.model, self.data, jacp, None, self._ee_site_id)

    # J^T @ delta_pos → joint deltas
    joint_delta = jacp[:, :7].T @ delta_pos

    # Position control: current_pos + delta
    self.data.ctrl[:7] = self.data.qpos[:7] + joint_delta
    self.data.ctrl[7:9] = action[3] * 0.04  # Gripper
```

This is more physically meaningful than directly controlling joint positions and produces smoother end-effector trajectories.

---

## Extension Guide

### Adding a New Model

1. Create a new file in `simscaleai/models/`:

```python
# simscaleai/models/my_model.py
from simscaleai.models.registry import register_model

@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self, state_dim=20, action_dim=4, **kwargs):
        super().__init__()
        # ... build network ...

    def forward(self, batch):
        # batch["observations"]["state"], batch["actions"], etc.
        # Must return dict with "loss" key
        return {"loss": loss, "predicted_actions": actions}

    def predict(self, obs):
        return self.net(obs["state"])
```

2. Import it in `simscaleai/models/registry.py` → `_ensure_registered()`:

```python
def _ensure_registered():
    if not _MODEL_REGISTRY:
        from simscaleai.models import bc, vla, my_model  # Add here
```

3. Now usable: `ModelRegistry.create("my_model", state_dim=20)`

### Adding a New Environment

1. Create a new file in `simscaleai/sim/envs/`:

```python
# simscaleai/sim/envs/push_env.py
from simscaleai.sim.base_env import BaseRobotEnv, SimConfig

class PushEnv(BaseRobotEnv):
    def __init__(self, config=None, render_mode=None):
        if config is None:
            config = SimConfig(xml_path="path/to/push.xml")
        super().__init__(config, render_mode)

    def _build_observation_space(self): ...
    def _build_action_space(self): ...
    def _get_obs(self): ...
    def _get_reward(self): ...
    def _is_terminated(self): ...
    def _reset_task(self, rng): ...
```

2. Register in `simscaleai/sim/factory.py` → `_register_defaults()`:

```python
def _register_defaults():
    from simscaleai.sim.envs.reach_env import ReachEnv
    from simscaleai.sim.envs.pick_place_env import PickPlaceEnv
    from simscaleai.sim.envs.push_env import PushEnv  # Add here

    register_env("reach", ReachEnv)
    register_env("pick_place", PickPlaceEnv)
    register_env("push", PushEnv)  # Add here
```

3. Now usable: `make_env("push")`

### Adding a New Reward Function

```python
from simscaleai.rl.rewards.rewards import RewardFunction

class VelocityPenalty(RewardFunction):
    """Penalize high joint velocities."""
    def __init__(self, key="joint_vel", scale=0.1):
        self.key = key
        self.scale = scale

    def compute(self, obs, action, info):
        return -self.scale * float(np.linalg.norm(obs[self.key]))
```

### Adding a New Policy Head

```python
# simscaleai/models/policy_heads/gmm_head.py
class GMMActionHead(nn.Module):
    """Gaussian Mixture Model action head for multi-modal distributions."""
    def __init__(self, input_dim, action_dim, n_components=5):
        super().__init__()
        self.means = nn.Linear(input_dim, n_components * action_dim)
        self.log_stds = nn.Linear(input_dim, n_components * action_dim)
        self.logits = nn.Linear(input_dim, n_components)

    def forward(self, features, target_actions=None):
        # ... GMM loss and sampling ...
```

---

## Data Flow

### Training Pipeline

```
HDF5 File ──→ TrajectoryDataset ──→ DataLoader ──→ Trainer.train()
                                         │
                                         ▼
                                    model(batch)
                                         │
                                         ▼
                                   output["loss"]
                                         │
                                         ▼
                              loss.backward() → optimizer.step()
                                         │
                                         ▼
                               checkpoint save / WandB log
```

### RL Pipeline

```
┌──────────┐     action      ┌─────────────┐
│          │ ──────────────→  │             │
│  Policy  │                  │ Environment │
│  (Actor) │ ← obs, reward ─ │  (MuJoCo)   │
│          │                  │             │
└────┬─────┘                  └─────────────┘
     │
     ▼
┌──────────────┐     GAE      ┌────────────┐
│ RolloutBuffer│ ───────────→ │ PPO Update │
│              │              │ (10 epochs)│
└──────────────┘              └────────────┘
```

### Data Generation Pipeline

```
┌─────────────┐    reset     ┌─────────────┐
│   Policy    │ ───────────→ │ Environment │
│ (random /   │    step      │  (MuJoCo)   │
│  scripted)  │ ← obs ───── │     + DR    │
└──────┬──────┘              └─────────────┘
       │
       │  record (obs, action, reward)
       ▼
┌──────────────┐
│   HDF5 File  │
│  gzip-comp.  │
└──────────────┘
```

---

## Threading & Concurrency

- **Physics stepping** is single-threaded (MuJoCo constraint)
- **DataLoader** uses multi-process workers (`num_workers`) for parallel data loading
- **DDP training** uses `torch.distributed` for gradient synchronization across processes
- **Rendering** uses OpenGL contexts (one per camera) — not thread-safe

---

## Performance Considerations

| Operation | Time (typical) | Notes |
|-----------|---------------|-------|
| `mj_step` | ~0.1ms | Single physics step at 500Hz |
| `env.step` | ~2.5ms | 25 substeps × 0.1ms |
| `render_camera` (128×128 RGB) | ~1ms | OpenGL offscreen rendering |
| BC forward (CPU, batch=32) | ~5ms | State-only, hidden=256 |
| VLA forward (CPU, batch=4) | ~50ms | image_size=64, embed_dim=64 |
| PPO update (2048 steps) | ~100ms | 10 epochs, batch=64 |

### Bottlenecks & Tips

1. **Rendering is optional** — skip camera obs during RL to save ~40% time per step
2. **Use small images** — resize to 64×64 for training, 128×128 for eval
3. **AMP cuts memory ~50%** — enables larger batch sizes
4. **DDP scales linearly** — 4 GPUs ≈ 4× throughput with near-linear scaling
5. **HDF5 is I/O bound** — use SSD storage, increase `num_workers`
