# Getting Started

## Installation

### Prerequisites

- Python ≥ 3.10
- pip or uv package manager
- (Optional) NVIDIA GPU with CUDA for accelerated training

### Basic Install

```bash
# Clone the repository
git clone https://github.com/yourusername/SimScaleAI.git
cd SimScaleAI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with all dependencies
pip install -e ".[all]"
```

### Install Options

```bash
# Minimal (simulation + training + models)
pip install -e .

# Development (adds pytest, ruff, mypy, pre-commit)
pip install -e ".[dev]"

# RL extras (adds stable-baselines3)
pip install -e ".[rl]"

# Transformers extras (adds HuggingFace transformers, timm)
pip install -e ".[transformers]"

# Everything
pip install -e ".[all]"
```

### Verify Installation

```bash
# Run the test suite
pytest tests/ -v

# Check CLI is accessible
simscale --help

# List available environments and models
simscale list-envs
simscale list-models
```

---

## Quick Start

### 1. Explore the Simulation

```python
from simscaleai.sim import make_env

# Create a reach environment
env = make_env("reach")
obs, info = env.reset(seed=42)

print(f"Observation keys: {obs.keys()}")
print(f"Joint positions: {obs['joint_pos'].shape}")  # (7,)
print(f"End-effector pos: {obs['ee_pos']}")           # (3,)
print(f"Target position:  {obs['target_pos']}")       # (3,)

# Step with a random action
action = env.action_space.sample()  # (4,) → 3D EE delta + gripper
obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward:.3f}, Distance: {info['distance']:.3f}")

env.close()
```

### 2. Generate Training Data

```bash
# Generate 100 episodes of reach demonstrations
simscale datagen --env-name reach --n-episodes 100 --output data/reach.h5
```

Or via Python:

```python
from simscaleai.datagen.generator import generate_dataset

stats = generate_dataset(
    env_name="reach",
    n_episodes=100,
    output_path="data/reach.h5",
    policy_type="scripted",   # P-controller expert
    domain_randomization=True,
    seed=42,
)
print(f"Generated {stats['total_episodes']} episodes, "
      f"{stats['total_steps']} steps, "
      f"success rate: {stats['success_rate']:.1%}")
```

### 3. Train a Model

```bash
# Train Behavior Cloning on the generated data
simscale train --model bc --dataset data/reach.h5 --max-steps 1000

# Train a VLA model with dummy data (for testing)
simscale train --model vla --max-steps 500
```

Or via Python:

```python
from torch.utils.data import DataLoader
from simscaleai.models import ModelRegistry
from simscaleai.training.trainer import Trainer, TrainConfig
from simscaleai.training.data.dataset import TrajectoryDataset

# Create model and dataset
model = ModelRegistry.create("bc", state_dim=20, action_dim=4)
dataset = TrajectoryDataset("data/reach.h5")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Configure and run training
config = TrainConfig(model_name="bc", max_steps=1000, lr=1e-4)
trainer = Trainer(model, loader, config)
metrics = trainer.train()
```

### 4. Train an RL Agent

```bash
# Train PPO in the reach environment
simscale rl --env-name reach --total-steps 50000
```

```python
from simscaleai.sim import make_env
from simscaleai.rl.agents.ppo import PPOAgent, PPOConfig

env = make_env("reach")
config = PPOConfig(total_timesteps=50_000, lr=3e-4)
agent = PPOAgent(obs_dim=20, action_dim=4, config=config)
history = agent.train(env)

# Save the trained agent
agent.save("checkpoints/ppo_reach.pt")
```

### 5. Evaluate a Policy

```bash
# Evaluate a checkpoint in simulation
simscale eval checkpoints/final.pt --env-name reach --n-episodes 20
```

```python
from simscaleai.sim import make_env
from simscaleai.rl.evaluator import evaluate_policy, EvalConfig

env = make_env("reach")
# ... load your model ...

results = evaluate_policy(
    env,
    predict_fn=model.predict,
    config=EvalConfig(n_episodes=20, deterministic=True),
)
print(f"Success rate: {results['success_rate']:.1%}")
print(f"Mean reward: {results['mean_reward']:.2f}")
```

---

## Running on Different Hardware

### CPU (any platform)

Everything works out of the box on CPU. Use small model configs for fast iteration:

```python
model = ModelRegistry.create("vla",
    image_size=64, embed_dim=64, num_heads=2, num_layers=2
)
```

### Apple Silicon (MPS)

PyTorch MPS backend is auto-detected. The trainer will use MPS when available:

```python
config = TrainConfig(device="auto")  # Auto-detects MPS
# Or explicitly:
config = TrainConfig(device="mps")
```

### NVIDIA GPU (CUDA)

For full-scale training with distributed support:

```python
config = TrainConfig(
    device="auto",       # Detects CUDA
    distributed=True,    # Enable DDP
    use_amp=True,        # Mixed precision
    amp_dtype="bfloat16",
    batch_size=256,
)
```

Launch distributed training:

```bash
torchrun --nproc_per_node=4 -m simscaleai.tools.cli train \
    --model vla --max-steps 100000 --batch-size 256
```

---

## Project Structure

```
SimScaleAI/
├── simscaleai/
│   ├── __init__.py             # Package root (version 0.1.0)
│   ├── sim/                    # Simulation environments
│   │   ├── base_env.py         # Abstract MuJoCo environment
│   │   ├── factory.py          # Environment registry & factory
│   │   ├── assets/             # MJCF robot/scene files (auto-generated)
│   │   └── envs/
│   │       ├── reach_env.py    # Reach task
│   │       └── pick_place_env.py # Pick-and-place task
│   ├── training/               # ML training infrastructure
│   │   ├── trainer.py          # Distributed training loop
│   │   └── data/
│   │       └── dataset.py      # HDF5 trajectory datasets
│   ├── models/                 # Foundation model architectures
│   │   ├── registry.py         # Model registry
│   │   ├── bc.py               # Behavior Cloning
│   │   ├── vla.py              # Vision-Language-Action
│   │   └── policy_heads/
│   │       ├── mlp_head.py     # MLP action head
│   │       └── diffusion_head.py # Diffusion Policy head
│   ├── rl/                     # Reinforcement learning
│   │   ├── evaluator.py        # Closed-loop evaluation
│   │   ├── agents/
│   │   │   └── ppo.py          # PPO with GAE
│   │   └── rewards/
│   │       └── rewards.py      # Composable reward functions
│   ├── datagen/                # Synthetic data generation
│   │   └── generator.py        # Dataset generation pipeline
│   └── tools/
│       └── cli.py              # Typer CLI entry point
├── tests/                      # Unit & integration tests
├── docs/                       # Documentation
├── .github/workflows/ci.yml    # CI pipeline
├── pyproject.toml              # Package config & dependencies
└── README.md
```
