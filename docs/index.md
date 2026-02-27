# SimScaleAI Documentation

**End-to-end robotics AI training and simulation platform** — from physics simulation to foundation model training, reinforcement learning, and deployment evaluation.

---

## Overview

SimScaleAI is a modular robotics AI infrastructure platform that demonstrates the full stack required for building, training, and evaluating robotic manipulation policies. It integrates:

- **MuJoCo 3.x physics simulation** with Gymnasium-compatible environments
- **Foundation model architectures** — Behavior Cloning (BC), Vision-Language-Action (VLA), Diffusion Policy
- **Distributed training** with PyTorch DDP, mixed precision (AMP), checkpointing, and WandB logging
- **Reinforcement learning** — PPO with GAE, composable reward functions, closed-loop evaluation
- **Synthetic data generation** — automated trajectory collection with domain randomization
- **CLI tooling** — train, evaluate, generate data, and explore components from the terminal

```
┌──────────────────────────────────────────────────────────┐
│                     SimScaleAI CLI                       │
│            simscale train | eval | datagen | rl          │
├──────────────┬───────────────┬───────────────────────────┤
│  Simulation  │    Training   │       Models              │
│  (MuJoCo)    │  Infrastructure│  (BC, VLA, Diffusion)    │
│              │  (PyTorch DDP) │                          │
│  • Reach     │  • Distributed │  • Behavior Cloning      │
│  • PickPlace │  • AMP/FSDP   │  • Vision-Language-Action │
│  • Domain    │  • Checkpoint  │  • Diffusion Policy Head │
│    Randomize │  • WandB log  │  • Model Registry        │
├──────────────┼───────────────┼───────────────────────────┤
│      RL Pipeline             │   Synthetic Data Gen      │
│  • PPO Agent                 │  • Scene randomization    │
│  • GAE Advantages            │  • Multi-modal capture    │
│  • Closed-loop eval          │  • HDF5 export            │
│  • Reward function library   │  • Dataset statistics     │
└──────────────────────────────┴───────────────────────────┘
```

## Documentation Guide

| Document | Description |
|----------|-------------|
| [Getting Started](getting-started.md) | Installation, quick start, and first experiment |
| [Simulation](simulation.md) | MuJoCo environments, camera rendering, domain randomization |
| [Models](models.md) | BC, VLA, Diffusion Policy architectures and model registry |
| [Training](training.md) | Distributed training, datasets, checkpointing, mixed precision |
| [Reinforcement Learning](reinforcement-learning.md) | PPO agent, reward functions, closed-loop evaluation |
| [Data Generation](data-generation.md) | Synthetic dataset pipeline, HDF5 format, scripted policies |
| [Visualization](visualization.md) | Simulation rendering, dataset plots, training curves |
| [CLI Reference](cli-reference.md) | All CLI commands with options and examples |
| [API Reference](api-reference.md) | Complete class/method/function reference |
| [Architecture](architecture.md) | Design patterns, extension guide, system internals |
| [Contributing](contributing.md) | Development setup, testing, CI pipeline, code style |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Physics Simulation | MuJoCo 3.x |
| ML Framework | PyTorch 2.x |
| Environment Interface | Gymnasium |
| Experiment Config | Hydra / OmegaConf |
| Logging | WandB / TensorBoard |
| Data Format | HDF5 (h5py) |
| CLI | Typer + Rich |
| Testing | pytest |
| Linting | Ruff |
| CI/CD | GitHub Actions |

## Requirements

- Python ≥ 3.10
- macOS, Linux, or Windows (MuJoCo 3.x is cross-platform)
- GPU optional — all features run on CPU/MPS with small model configs
