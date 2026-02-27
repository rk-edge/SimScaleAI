# CLI Reference

SimScaleAI provides a command-line interface via the `simscale` command, built with [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/) for formatted output.

---

## Commands Overview

| Command | Description |
|---------|-------------|
| `simscale train` | Train a model on trajectory data |
| `simscale eval` | Evaluate a checkpoint in simulation |
| `simscale datagen` | Generate synthetic trajectory datasets |
| `simscale rl` | Train a PPO agent in simulation |
| `simscale list-envs` | List all registered environments |
| `simscale list-models` | List all registered models |

---

## `simscale train`

Train a model on trajectory data with distributed training support.

```bash
simscale train [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | `str` | `"bc"` | Model architecture name |
| `--dataset` | `str` | `""` | Path to HDF5 dataset (empty = dummy data) |
| `--batch-size` | `int` | `32` | Training batch size |
| `--max-steps` | `int` | `1000` | Maximum training steps |
| `--lr` | `float` | `1e-4` | Learning rate |
| `--checkpoint-dir` | `str` | `"checkpoints"` | Checkpoint save directory |
| `--use-wandb` | flag | `False` | Enable WandB logging |
| `--device` | `str` | `"auto"` | Device: `auto`, `cuda`, `mps`, `cpu` |
| `-v` / `--verbose` | flag | `False` | Verbose output |

### Examples

```bash
# Train BC on real data
simscale train --model bc --dataset data/reach.h5 --max-steps 5000

# Train VLA with dummy data (quick test)
simscale train --model vla --max-steps 500

# Train with WandB logging
simscale train --model bc --dataset data/reach.h5 \
    --max-steps 50000 --use-wandb --batch-size 64

# Train on specific device
simscale train --model bc --device cpu --max-steps 1000

# Verbose output
simscale train --model bc -v
```

---

## `simscale eval`

Evaluate a trained model checkpoint in closed-loop simulation.

```bash
simscale eval CHECKPOINT [OPTIONS]
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `CHECKPOINT` | `str` | Path to model checkpoint file |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--env-name` | `str` | `"reach"` | Environment name |
| `--n-episodes` | `int` | `20` | Number of evaluation episodes |
| `--render` | flag | `False` | Render simulation to screen |

### Examples

```bash
# Evaluate latest checkpoint
simscale eval checkpoints/final.pt --env-name reach

# Evaluate with more episodes
simscale eval checkpoints/step_10000.pt --n-episodes 100

# Evaluate with visual rendering
simscale eval checkpoints/final.pt --render

# Evaluate on pick-and-place
simscale eval checkpoints/final.pt --env-name pick_place --n-episodes 50
```

### Output

Displays a Rich table with evaluation metrics:

```
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric        ┃ Value      ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ success_rate  │ 0.85       │
│ mean_reward   │ -12.34     │
│ std_reward    │ 5.67       │
│ mean_length   │ 145.2      │
│ min_reward    │ -25.10     │
│ max_reward    │ -2.50      │
└───────────────┴────────────┘
```

---

## `simscale datagen`

Generate synthetic trajectory datasets from simulation.

```bash
simscale datagen [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--env-name` | `str` | `"reach"` | Environment name |
| `--n-episodes` | `int` | `100` | Number of episodes to collect |
| `--output` | `str` | `"data/dataset.h5"` | Output HDF5 file path |
| `--policy` | `str` | `"random"` | Policy type: `random` or `scripted` |
| `--randomize` | flag | `False` | Enable domain randomization |

### Examples

```bash
# Quick random dataset
simscale datagen --env-name reach --n-episodes 50

# Expert demonstrations with domain randomization
simscale datagen --env-name reach --n-episodes 1000 \
    --output data/reach_expert.h5 --policy scripted --randomize

# Pick-and-place data
simscale datagen --env-name pick_place --n-episodes 500 \
    --output data/pick_place.h5 --policy random

# Large-scale dataset
simscale datagen --env-name reach --n-episodes 10000 \
    --output data/reach_10k.h5 --policy scripted --randomize
```

### Output

Displays a Rich panel with dataset statistics:

```
╭────────── Dataset Generation Results ──────────╮
│ Total episodes:      100                       │
│ Total steps:         18,542                    │
│ Mean episode length: 185.4                     │
│ Mean reward:         -8.72                     │
│ Success rate:        72.0%                     │
│ File size:           12.3 MB                   │
│ Output path:         data/reach_expert.h5      │
╰────────────────────────────────────────────────╯
```

---

## `simscale rl`

Train a PPO reinforcement learning agent in simulation.

```bash
simscale rl [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--env-name` | `str` | `"reach"` | Environment name |
| `--total-steps` | `int` | `100000` | Total environment steps |
| `--lr` | `float` | `3e-4` | Learning rate |
| `--save-path` | `str` | `"checkpoints/ppo_agent.pt"` | Save path for trained agent |

### Examples

```bash
# Quick RL training
simscale rl --env-name reach --total-steps 50000

# Longer training
simscale rl --env-name reach --total-steps 1000000 --lr 1e-4

# Pick-and-place RL
simscale rl --env-name pick_place --total-steps 500000

# Custom save path
simscale rl --env-name reach --save-path models/ppo_v2.pt
```

---

## `simscale list-envs`

List all registered simulation environments.

```bash
simscale list-envs
```

### Output

```
╭──────── Available Environments ────────╮
│ • reach                                │
│ • pick_place                           │
╰────────────────────────────────────────╯
```

---

## `simscale list-models`

List all registered model architectures.

```bash
simscale list-models
```

### Output

```
╭──────── Available Models ────────╮
│ • bc                             │
│ • vla                            │
╰──────────────────────────────────╯
```

---

## Global Options

```bash
# Show help for any command
simscale --help
simscale train --help
simscale eval --help

# Show version
simscale --version
```

---

## Shell Completion

Typer supports shell completion for bash, zsh, and fish:

```bash
# Generate completion script
simscale --install-completion

# Or manually for zsh
eval "$(simscale --show-completion zsh)"
```
