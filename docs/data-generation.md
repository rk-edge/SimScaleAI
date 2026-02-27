# Data Generation

SimScaleAI includes a synthetic data generation pipeline that collects robot trajectories from simulation, optionally with domain randomization, and exports them as HDF5 datasets for training.

---

## Overview

The data generation pipeline:

1. Instantiates a simulation environment
2. Runs a policy (random or scripted) for N episodes
3. Records observations, actions, and rewards at each timestep
4. Exports everything to a compressed HDF5 file
5. Reports dataset statistics

```
Environment ──→ Policy ──→ Trajectory ──→ HDF5 Dataset
    │              │           │
    │   random /   │  obs, actions,    ├── episode_0/
    │   scripted   │  rewards          │   ├── observations/
    │              │                   │   ├── actions
    └──────────────┘                   │   └── rewards
                                       └── episode_N/
```

---

## CLI Usage

```bash
# Generate 100 episodes with random policy
simscale datagen --env-name reach --n-episodes 100 --output data/reach_random.h5

# Generate with scripted expert policy
simscale datagen --env-name reach --n-episodes 500 \
    --output data/reach_expert.h5 --policy scripted

# With domain randomization
simscale datagen --env-name reach --n-episodes 1000 \
    --output data/reach_dr.h5 --policy scripted --randomize

# Pick-and-place data
simscale datagen --env-name pick_place --n-episodes 200 \
    --output data/pick_place.h5 --policy random
```

---

## Python API

```python
from simscaleai.datagen.generator import generate_dataset

stats = generate_dataset(
    env_name="reach",               # Environment name
    n_episodes=100,                 # Number of episodes to collect
    output_path="data/reach.h5",    # Output HDF5 file path
    policy_type="scripted",         # "random" or "scripted"
    domain_randomization=True,      # Enable domain randomization
    max_steps=200,                  # Max steps per episode
    seed=42,                        # Random seed for reproducibility
)

print(f"Episodes:      {stats['total_episodes']}")
print(f"Total steps:   {stats['total_steps']}")
print(f"Mean length:   {stats['mean_episode_length']:.1f}")
print(f"Mean reward:   {stats['mean_reward']:.2f}")
print(f"Success rate:  {stats['success_rate']:.1%}")
print(f"File size:     {stats['file_size_mb']:.1f} MB")
print(f"Output path:   {stats['output_path']}")
```

### Return Values

| Metric | Type | Description |
|--------|------|-------------|
| `total_episodes` | `int` | Number of episodes collected |
| `total_steps` | `int` | Total timesteps across all episodes |
| `mean_episode_length` | `float` | Average episode length |
| `mean_reward` | `float` | Average total episode reward |
| `success_rate` | `float` | Fraction of episodes with `info["success"]` |
| `output_path` | `str` | Path to the generated HDF5 file |
| `file_size_mb` | `float` | File size in megabytes |

---

## Policies

### Random Policy

Samples uniformly from the action space. Useful for generating diverse exploration data:

```python
stats = generate_dataset(
    env_name="reach",
    n_episodes=100,
    policy_type="random",
)
```

### Scripted Reach Policy

A proportional (P-controller) expert that moves the end-effector toward the target:

```python
stats = generate_dataset(
    env_name="reach",
    n_episodes=500,
    policy_type="scripted",
)
```

**Algorithm:**

```
direction = normalize(target_pos - ee_pos)
action[:3] = direction * 0.5     # Move toward target
action[3]  = 0.0                 # Gripper open
```

This produces smooth, goal-directed trajectories with high success rates — ideal for behavior cloning training data.

---

## HDF5 Output Format

The generated HDF5 files follow the structure expected by `TrajectoryDataset`:

```
dataset.h5
├── episode_0/
│   ├── observations/
│   │   ├── joint_pos     (T, 7)    float64   # Joint angles
│   │   ├── joint_vel     (T, 7)    float64   # Joint velocities
│   │   ├── ee_pos        (T, 3)    float64   # End-effector position
│   │   ├── target_pos    (T, 3)    float64   # Target position
│   │   └── image         (T, H, W, 3) uint8  # Camera image (if cameras enabled)
│   ├── actions           (T, 4)    float64   # Robot actions
│   └── rewards           (T,)      float64   # Per-step rewards
├── episode_1/
│   └── ...
└── episode_N/
    └── ...
```

All datasets use **gzip compression** for efficient storage.

### Reading Generated Data

```python
import h5py

with h5py.File("data/reach.h5", "r") as f:
    print(f"Number of episodes: {len(f.keys())}")

    ep = f["episode_0"]
    print(f"Episode length: {ep['actions'].shape[0]}")
    print(f"Action dim: {ep['actions'].shape[1]}")
    print(f"Obs keys: {list(ep['observations'].keys())}")
```

Or use the built-in dataset class:

```python
from simscaleai.training.data.dataset import TrajectoryDataset
from torch.utils.data import DataLoader

dataset = TrajectoryDataset("data/reach.h5")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    print(batch["observations"]["joint_pos"].shape)  # (32, 7)
    print(batch["actions"].shape)                     # (32, 4)
    break
```

---

## Domain Randomization

When `domain_randomization=True`, each episode reset applies random variations:

| Parameter | Randomization | Purpose |
|-----------|---------------|---------|
| Light direction | Uniform on unit sphere | Visual diversity for camera observations |
| Surface friction | ±20% of defaults | Physical diversity for contact behavior |

This produces more robust training data, especially for vision-based policies.

---

## Scaling Data Generation

### Large-Scale Collection

```python
# Generate a large dataset with many episodes
stats = generate_dataset(
    env_name="reach",
    n_episodes=10_000,
    output_path="data/reach_10k.h5",
    policy_type="scripted",
    domain_randomization=True,
    seed=42,
)
```

### Multiple Datasets

Generate datasets for different tasks or conditions:

```python
for env_name in ["reach", "pick_place"]:
    for policy in ["random", "scripted"]:
        stats = generate_dataset(
            env_name=env_name,
            n_episodes=500,
            output_path=f"data/{env_name}_{policy}.h5",
            policy_type=policy,
            domain_randomization=True,
        )
        print(f"{env_name}/{policy}: {stats['success_rate']:.1%} success")
```

---

## Pipeline: Data Generation → Training

A typical workflow:

```bash
# 1. Generate expert demonstrations
simscale datagen --env-name reach --n-episodes 1000 \
    --output data/reach_expert.h5 --policy scripted --randomize

# 2. Train behavior cloning model
simscale train --model bc --dataset data/reach_expert.h5 \
    --max-steps 50000 --batch-size 64

# 3. Evaluate in simulation
simscale eval checkpoints/final.pt --env-name reach --n-episodes 100
```

```python
from simscaleai.datagen.generator import generate_dataset
from simscaleai.training.data.dataset import TrajectoryDataset
from simscaleai.training.trainer import Trainer, TrainConfig
from simscaleai.models import ModelRegistry
from torch.utils.data import DataLoader

# Generate
generate_dataset("reach", n_episodes=1000, output_path="data/reach.h5",
                 policy_type="scripted", domain_randomization=True)

# Load
dataset = TrajectoryDataset("data/reach.h5")
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Train
model = ModelRegistry.create("bc", state_dim=20, action_dim=4)
config = TrainConfig(max_steps=50_000, lr=1e-4)
trainer = Trainer(model, loader, config)
trainer.train()
```
