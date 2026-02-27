# Simulation

SimScaleAI uses **MuJoCo 3.x** as its physics backend, wrapped in **Gymnasium**-compatible environments. All environments follow the template method pattern: `BaseRobotEnv` defines the simulation loop, and task-specific subclasses implement observation, reward, and reset logic.

---

## Environment Architecture

```
BaseRobotEnv (abstract)
├── MuJoCo model loading & physics stepping
├── Camera rendering (RGB, depth, segmentation)
├── Domain randomization hooks
├── Clean observation/action space interface
│
├── ReachEnv
│   └── Move end-effector to target position
│
└── PickPlaceEnv
    └── Grasp object and place at target location
```

---

## Configuration

### SimConfig

All environments are configured through `SimConfig`:

```python
from simscaleai.sim.base_env import SimConfig, CameraConfig

config = SimConfig(
    xml_path="",                    # MJCF file (auto-generated if empty)
    dt=0.002,                       # Physics timestep (500 Hz)
    control_dt=0.05,                # Control timestep (20 Hz)
    max_episode_steps=200,          # Episode length limit
    cameras=[CameraConfig()],       # Camera configuration list
    domain_randomization=False,     # Enable DR
    seed=None,                      # Random seed
)
```

The ratio `control_dt / dt` determines how many physics substeps run per `env.step()` call. Default is 25 substeps (0.05 / 0.002).

### CameraConfig

Each camera is configured independently:

```python
camera = CameraConfig(
    name="wrist_cam",           # Camera name in MJCF XML
    width=128,                  # Image width (px)
    height=128,                 # Image height (px)
    fov=60.0,                   # Field of view (degrees)
    render_depth=True,          # Include depth map
    render_segmentation=False,  # Include segmentation mask
)
```

---

## Available Environments

### Reach (`reach`)

The simplest manipulation task — move the Franka Panda's end-effector to a random target position.

| Property | Details |
|----------|---------|
| **Robot** | Franka Panda 7-DOF arm + 2-finger gripper |
| **Task** | Move end-effector to target sphere |
| **Observation** | `joint_pos (7,)`, `joint_vel (7,)`, `ee_pos (3,)`, `target_pos (3,)`, optional `image (H,W,3)` |
| **Action** | `(4,)` — 3D end-effector delta (x,y,z) + gripper command |
| **Reward** | `-L2(ee, target)` + `+1.0` bonus if distance < 0.05m |
| **Termination** | Distance < 0.02m (success) |
| **Target workspace** | x ∈ [0.3, 0.7], y ∈ [-0.3, 0.3], z ∈ [0.1, 0.5] |

```python
from simscaleai.sim import make_env

env = make_env("reach")
obs, info = env.reset(seed=42)

for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: dist={info['distance']:.3f}, reward={reward:.3f}")
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

**Action mapping:** The 4D action is mapped to the 9 MuJoCo actuators (7 arm joints + 2 gripper fingers) using **Jacobian-transpose inverse kinematics**:

1. Compute positional Jacobian at end-effector site via `mj_jacSite`
2. Multiply `J_arm^T @ delta_pos` to get joint-space deltas
3. Add deltas to current joint positions for position control
4. Map gripper command to symmetric finger positions

### Pick and Place (`pick_place`)

A multi-stage manipulation task — grasp an object and place it at a target location.

| Property | Details |
|----------|---------|
| **Robot** | Franka Panda 7-DOF arm + 2-finger gripper |
| **Task** | Pick up cube, place at target position |
| **Observation** | `joint_pos (7,)`, `joint_vel (7,)`, `ee_pos (3,)`, `object_pos (3,)`, `object_quat (4,)`, `target_pos (3,)`, `gripper_state (1,)`, optional `image` |
| **Action** | `(4,)` — 3D end-effector delta + gripper |
| **Reward** | 4-stage shaped (see below) |
| **Termination** | Object lifted AND placed within 0.03m |

**Multi-stage reward shaping:**

| Stage | Condition | Reward |
|-------|-----------|--------|
| 1. Reach | Always | `-L2(ee, object)` |
| 2. Grasp | `reach_dist < 0.05` | `+0.5` bonus |
| 3. Lift | `object_z > 0.55` | `+0.5` bonus |
| 4. Place | Object lifted | `-L2(object, target)` + `+2.0` if placed |

```python
env = make_env("pick_place")
obs, info = env.reset(seed=42)

# Info dict tracks progress
# info = {"distance", "reach_dist", "place_dist", "grasped", "lifted", "success"}
```

---

## Camera Rendering

Environments support multi-modal camera rendering:

```python
from simscaleai.sim.base_env import SimConfig, CameraConfig

config = SimConfig(
    cameras=[
        CameraConfig(name="wrist_cam", width=256, height=256,
                      render_depth=True, render_segmentation=True),
    ]
)

env = make_env("reach", config=config)
obs, _ = env.reset()

# Image is included in observations when cameras are configured
rgb_image = obs["image"]  # (256, 256, 3) uint8
```

Direct camera rendering:

```python
# Render specific camera
camera_data = env.render_camera(
    camera_name="wrist_cam",
    depth=True,
    segmentation=True,
)

rgb = camera_data["rgb"]              # (H, W, 3) uint8
depth = camera_data["depth"]          # (H, W) float32
segmentation = camera_data["segmentation"]  # (H, W) int32
```

---

## Domain Randomization

When enabled, domain randomization is applied at each `reset()`:

```python
config = SimConfig(domain_randomization=True)
env = make_env("reach", config=config)
```

**Randomized parameters:**

| Parameter | Range | Method |
|-----------|-------|--------|
| Light direction | Uniform on unit sphere | `model.light_dir[0]` |
| Surface friction | ±20% of defaults | `model.geom_friction[:, 0]` |

To extend domain randomization, override `_apply_domain_randomization(rng)` in your environment subclass.

---

## Environment Factory

The factory provides a clean interface for creating environments by name:

```python
from simscaleai.sim.factory import make_env, list_envs, register_env

# List registered environments
print(list_envs())  # ['pick_place', 'reach']

# Create environment
env = make_env("reach", render_mode="rgb_array")

# Pass config kwargs
env = make_env("reach", max_episode_steps=500, domain_randomization=True)
```

### Registering Custom Environments

```python
from simscaleai.sim.base_env import BaseRobotEnv, SimConfig
from simscaleai.sim.factory import register_env

class MyCustomEnv(BaseRobotEnv):
    def _build_observation_space(self): ...
    def _build_action_space(self): ...
    def _get_obs(self): ...
    def _get_reward(self): ...
    def _is_terminated(self): ...
    def _reset_task(self, rng): ...

register_env("my_task", MyCustomEnv)
env = make_env("my_task")
```

---

## Creating Custom Environments

Subclass `BaseRobotEnv` and implement 6 abstract methods:

```python
import numpy as np
from gymnasium import spaces
from simscaleai.sim.base_env import BaseRobotEnv, SimConfig

class PushEnv(BaseRobotEnv):
    """Push an object to a target location."""

    def __init__(self, config=None, render_mode=None):
        if config is None:
            config = SimConfig(xml_path="path/to/push_scene.xml")
        super().__init__(config, render_mode)

    def _build_observation_space(self) -> spaces.Space:
        return spaces.Dict({
            "joint_pos": spaces.Box(-np.pi, np.pi, shape=(7,)),
            "object_pos": spaces.Box(-2.0, 2.0, shape=(3,)),
            "target_pos": spaces.Box(-2.0, 2.0, shape=(3,)),
        })

    def _build_action_space(self) -> spaces.Space:
        return spaces.Box(-1.0, 1.0, shape=(3,))

    def _get_obs(self) -> dict:
        return {
            "joint_pos": self.data.qpos[:7].copy(),
            "object_pos": self.data.body("object").xpos.copy(),
            "target_pos": self._target_pos.copy(),
        }

    def _get_reward(self) -> float:
        obj_pos = self.data.body("object").xpos
        return -float(np.linalg.norm(obj_pos - self._target_pos))

    def _is_terminated(self) -> bool:
        obj_pos = self.data.body("object").xpos
        return bool(np.linalg.norm(obj_pos - self._target_pos) < 0.03)

    def _reset_task(self, rng: np.random.Generator) -> None:
        self._target_pos = rng.uniform([-0.3, -0.3, 0.4], [0.3, 0.3, 0.4])
```

### BaseRobotEnv Lifecycle

```
__init__()
  └── Load MuJoCo model from XML
  └── Create renderers for each camera
  └── Call _build_observation_space()
  └── Call _build_action_space()

reset(seed=...)
  └── Reset MuJoCo state (qpos/qvel to defaults)
  └── Apply domain randomization (if enabled)
  └── Call _reset_task(rng) — set up new episode
  └── Step physics once for initial state
  └── Return (_get_obs(), _get_info())

step(action)
  └── _apply_action(action) — clips and sets ctrl
  └── mj_step() × n_substeps
  └── Return (_get_obs(), _get_reward(), _is_terminated(), truncated, _get_info())
```

---

## MJCF Scene Files

Environments auto-generate MJCF XML files on first use. These are saved to `simscaleai/sim/assets/`:

- `panda_reach.xml` — Panda arm + table + target sphere + 2 cameras + 9 actuators
- `panda_pick_place.xml` — Panda arm + table + manipulable cube + placement target

The auto-generated scenes include:

- **Franka Panda arm**: 7 revolute joints modeled with accurate link lengths and mass properties
- **Parallel-jaw gripper**: 2 prismatic joints for finger control
- **Position actuators**: `kp=100, kv=20` for stable position control
- **Cameras**: `wrist_cam` (on end-effector) and `overhead_cam` (top-down view)
- **Contact properties**: `condim=4`, `friction=1.0 0.5 0.01`
- **Solver**: `implicitfast` integrator at 500 Hz

You can provide your own MJCF files via `SimConfig(xml_path="my_scene.xml")`.
