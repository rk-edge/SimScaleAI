# Visualization

SimScaleAI includes built-in visualization tools for inspecting simulation environments, datasets, training progress, and RL metrics. All visualizations are available via CLI commands and the Python API.

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `simscale viz-env` | Render a grid of simulation frames from a rollout |
| `simscale viz-cameras` | Show RGB, depth, and segmentation side-by-side |
| `simscale viz-dataset` | Plot dataset statistics (episode lengths, rewards, action distributions) |
| `simscale viz-trajectory` | Plot a single episode timeline (observations, actions, rewards) |
| `simscale viz-live` | Launch the interactive 3D MuJoCo viewer |

All commands support `--save path.png` to export images and `--help` for full options.

---

## Simulation Visualization

### Environment Frame Grid

Renders an environment rollout as a grid of camera frames, showing how the scene evolves over time.

```bash
# Display interactively
simscale viz-env --env-name reach --n-steps 20

# Save to file
simscale viz-env --env-name reach --n-steps 20 --save viz/env_grid.png

# Pick-and-place environment
simscale viz-env --env-name pick_place --n-steps 30 --save viz/pick_place.png
```

```python
from simscaleai.tools.visualize import render_env_grid

render_env_grid(
    env_name="reach",
    n_steps=20,          # Steps to simulate
    seed=42,             # Reproducible
    save_path="env.png", # Save to file (optional)
    show=True,           # Display interactively
)
```

### Camera Modalities

Shows all camera outputs side-by-side: RGB color, depth map (with colorbar), and segmentation mask.

```bash
simscale viz-cameras --env-name reach --save viz/cameras.png
```

```python
from simscaleai.tools.visualize import render_camera_modalities

render_camera_modalities(
    env_name="reach",
    seed=42,
    save_path="cameras.png",
    show=True,
)
```

**Output panels:**

| Panel | Format | Description |
|-------|--------|-------------|
| RGB | `(H, W, 3)` uint8 | Color image from wrist camera |
| Depth | `(H, W)` float32 | Distance map with viridis colormap |
| Segmentation | `(H, W)` int32 | Object ID mask with tab20 colormap |

### Interactive 3D Viewer

Launches the MuJoCo passive viewer for real-time observation. The robot executes random actions.

```bash
simscale viz-live --env-name reach --n-episodes 3
```

```python
from simscaleai.tools.visualize import run_interactive

run_interactive(
    env_name="reach",
    n_episodes=3,
    max_steps=200,
    seed=42,
)
```

---

## Dataset Visualization

### Dataset Statistics

Plots distributions from an HDF5 trajectory dataset: episode lengths, rewards, and per-dimension action histograms.

```bash
# First generate some data
simscale datagen --env-name reach --n-episodes 100 --output data/reach.h5 --policy scripted

# Then visualize
simscale viz-dataset data/reach.h5 --save viz/dataset_stats.png
```

```python
from simscaleai.tools.visualize import plot_dataset_stats

stats = plot_dataset_stats(
    data_path="data/reach.h5",
    save_path="stats.png",
    show=True,
)

# stats = {
#     "n_episodes": 100,
#     "mean_episode_length": 185.4,
#     "std_episode_length": 12.3,
#     "total_steps": 18540,
#     "action_dim": 4,
#     "action_mean": [0.01, -0.02, 0.03, 0.0],
#     "action_std": [0.28, 0.31, 0.25, 0.29],
#     "mean_reward": -8.72,
#     "std_reward": 3.14,
# }
```

**Plot layout:**

| Row | Panels | Content |
|-----|--------|---------|
| Top | 1–2 | Episode length histogram + reward distribution (with mean lines) |
| Bottom | 3–6 | Action dimension histograms (one per action dim, up to 4) |

### Single Trajectory

Plots one episode over time — observations, actions, and rewards as time series.

```bash
simscale viz-trajectory data/reach.h5 --episode 0 --save viz/trajectory.png
```

```python
from simscaleai.tools.visualize import plot_trajectory

plot_trajectory(
    data_path="data/reach.h5",
    episode_idx=0,          # Which episode to plot
    save_path="traj.png",
    show=True,
)
```

**Plot panels** (one per observation key + actions + rewards):

- `joint_pos` — 7 lines (one per joint angle)
- `joint_vel` — 7 lines
- `ee_pos` — 3 lines (x, y, z)
- `target_pos` — 3 lines (x, y, z)
- Actions — 4 lines (delta_x, delta_y, delta_z, gripper)
- Reward — single line with shaded area

Image observations are automatically skipped (only scalar/vector observations are plotted).

---

## Training Visualization

### Training Metrics

Plot loss curves, learning rate schedules, or any dict of metric lists returned by the trainer.

```python
from simscaleai.tools.visualize import plot_training_metrics

# After training
# metrics = trainer.train()

# Or manually construct
metrics = {
    "loss": [2.1, 1.8, 1.5, 1.2, ...],
    "learning_rate": [1e-5, 2e-5, 3e-5, ...],
}

plot_training_metrics(
    metrics,
    title="BC Training on Reach Data",
    save_path="training.png",
    show=True,
)
```

**Features:**

- One panel per metric (stacked vertically, shared x-axis)
- Raw values shown with low opacity
- Smoothed line overlay (rolling average) when >20 data points
- Automatic smoothing window size

### RL Training Progress

Plot reward curves, episode lengths, and policy/value losses from PPO training.

```python
from simscaleai.tools.visualize import plot_rl_training

# After RL training
# history = agent.train(env)

history = {
    "episode_reward": [-100, -95, -88, ...],
    "episode_length": [200, 195, 180, ...],
    "policy_loss": [0.05, 0.04, 0.03, ...],
    "value_loss": [0.8, 0.6, 0.5, ...],
}

plot_rl_training(
    history,
    title="PPO on Reach Environment",
    save_path="rl_progress.png",
    show=True,
)
```

**Panels** (only shown if data exists):

| Key | Label | Color |
|-----|-------|-------|
| `episode_reward` | Episode Reward | Green |
| `episode_length` | Episode Length | Blue |
| `policy_loss` | Policy Loss | Red |
| `value_loss` | Value Loss | Orange |

Each panel shows raw data (faded) with a rolling average overlay.

---

## Saving & Export

All visualization functions accept `save_path` and `show` parameters:

```python
# Save only (no display) — useful in scripts and CI
render_env_grid("reach", save_path="output.png", show=False)

# Display only (no save) — interactive exploration
render_env_grid("reach", show=True)

# Both
render_env_grid("reach", save_path="output.png", show=True)
```

Images are saved at **150 DPI** with tight bounding boxes. Parent directories are created automatically.

---

## Python API Reference

```python
from simscaleai.tools.visualize import (
    # Simulation
    render_env_grid,           # Frame grid of environment rollout
    render_camera_modalities,  # RGB / depth / segmentation panels
    run_interactive,           # Live 3D MuJoCo viewer

    # Datasets
    plot_dataset_stats,        # Episode/reward/action distributions
    plot_trajectory,           # Single episode timeline

    # Training
    plot_training_metrics,     # Loss curves, LR schedule, any metrics
    plot_rl_training,          # RL reward/loss progress curves
)
```

| Function | Parameters | Returns |
|----------|-----------|---------|
| `render_env_grid` | `env_name, n_steps=10, seed=42, save_path=None, show=True` | `None` |
| `render_camera_modalities` | `env_name, seed=42, save_path=None, show=True` | `None` |
| `run_interactive` | `env_name, n_episodes=3, max_steps=200, seed=42` | `None` |
| `plot_dataset_stats` | `data_path, save_path=None, show=True` | `dict[str, Any]` |
| `plot_trajectory` | `data_path, episode_idx=0, save_path=None, show=True` | `None` |
| `plot_training_metrics` | `metrics, title="...", save_path=None, show=True` | `None` |
| `plot_rl_training` | `history, title="...", save_path=None, show=True` | `None` |
