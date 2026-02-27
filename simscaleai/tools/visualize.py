"""Visualization tools for SimScaleAI.

Provides functions for visualizing:
- Simulation environments (RGB, depth, segmentation cameras)
- Training metrics (loss curves, learning rate schedules)
- Trajectory datasets (observation/action distributions, episode replays)
- RL training progress (reward curves, success rates)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _import_plt():
    """Lazy import matplotlib with Agg backend fallback."""
    import matplotlib

    try:
        import matplotlib.pyplot as plt

        # Test if a display is available
        fig = plt.figure()
        plt.close(fig)
    except Exception:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    return plt


# ── Simulation Visualization ──────────────────────────────────────────────


def render_env_grid(
    env_name: str = "reach",
    n_steps: int = 10,
    seed: int = 42,
    save_path: str | None = None,
    show: bool = True,
) -> np.ndarray | None:
    """Render a grid of simulation frames showing an environment rollout.

    Args:
        env_name: Environment name.
        n_steps: Number of steps to render (up to 8 shown in grid).
        seed: Random seed.
        save_path: Path to save image (PNG). None = don't save.
        show: Display the plot interactively.

    Returns:
        RGB image array if save_path is set, else None.
    """
    from simscaleai.sim.factory import make_env

    plt = _import_plt()

    env = make_env(env_name)
    obs, info = env.reset(seed=seed)

    frames = []
    step_indices = []

    # Capture initial frame
    cam_data = env.render_camera()
    frames.append(cam_data["rgb"])
    step_indices.append(0)

    for step in range(1, n_steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        cam_data = env.render_camera()
        frames.append(cam_data["rgb"])
        step_indices.append(step)
        if terminated or truncated:
            break

    env.close()

    # Select up to 8 frames evenly spaced
    n_show = min(len(frames), 8)
    indices = np.linspace(0, len(frames) - 1, n_show, dtype=int)
    selected_frames = [frames[i] for i in indices]
    selected_steps = [step_indices[i] for i in indices]

    cols = min(n_show, 4)
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, (frame, step) in enumerate(zip(selected_frames, selected_steps)):
        r, c = divmod(i, cols)
        axes[r, c].imshow(frame)
        axes[r, c].set_title(f"Step {step}", fontsize=10)
        axes[r, c].axis("off")

    # Hide unused axes
    for i in range(n_show, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle(f"Environment: {env_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved environment grid to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return None


def render_camera_modalities(
    env_name: str = "reach",
    seed: int = 42,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Show RGB, depth, and segmentation from the environment camera.

    Args:
        env_name: Environment name.
        seed: Random seed.
        save_path: Path to save image (PNG).
        show: Display the plot interactively.
    """
    from simscaleai.sim.base_env import CameraConfig, SimConfig
    from simscaleai.sim.factory import make_env

    plt = _import_plt()

    env = make_env(env_name)
    env.reset(seed=seed)

    # Step a few times for a non-trivial state
    for _ in range(5):
        env.step(env.action_space.sample())

    cam_data = env.render_camera(depth=True, segmentation=True)
    env.close()

    n_panels = len(cam_data)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    idx = 0
    if "rgb" in cam_data:
        axes[idx].imshow(cam_data["rgb"])
        axes[idx].set_title("RGB", fontsize=12)
        axes[idx].axis("off")
        idx += 1

    if "depth" in cam_data:
        im = axes[idx].imshow(cam_data["depth"], cmap="viridis")
        axes[idx].set_title("Depth", fontsize=12)
        axes[idx].axis("off")
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        idx += 1

    if "segmentation" in cam_data:
        seg = cam_data["segmentation"]
        if seg.ndim == 3:
            seg = seg[:, :, 0]  # Use first channel (object ID)
        axes[idx].imshow(seg, cmap="tab20")
        axes[idx].set_title("Segmentation", fontsize=12)
        axes[idx].axis("off")
        idx += 1

    fig.suptitle(f"Camera Modalities — {env_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ── Dataset Visualization ────────────────────────────────────────────────


def plot_dataset_stats(
    data_path: str,
    save_path: str | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """Visualize statistics of an HDF5 trajectory dataset.

    Plots:
    - Episode length distribution
    - Reward distribution
    - Action dimension distributions
    - Observation key distributions

    Args:
        data_path: Path to HDF5 dataset.
        save_path: Path to save image (PNG).
        show: Display the plot interactively.

    Returns:
        Dict with computed statistics.
    """
    import h5py

    plt = _import_plt()

    with h5py.File(data_path, "r") as f:
        episode_keys = sorted(f.keys())
        n_episodes = len(episode_keys)

        episode_lengths = []
        episode_rewards = []
        all_actions = []
        obs_stats: dict[str, list[np.ndarray]] = {}

        for ep_key in episode_keys:
            ep = f[ep_key]
            actions = ep["actions"][:]
            episode_lengths.append(len(actions))
            all_actions.append(actions)

            if "rewards" in ep:
                episode_rewards.append(float(np.sum(ep["rewards"][:])))

            if "observations" in ep:
                for obs_key in ep["observations"]:
                    if obs_key not in obs_stats:
                        obs_stats[obs_key] = []
                    data = ep["observations"][obs_key][:]
                    if data.ndim <= 2:  # Skip images
                        obs_stats[obs_key].append(data)

    all_actions_arr = np.concatenate(all_actions, axis=0)
    action_dim = all_actions_arr.shape[1]

    # Compute statistics
    stats = {
        "n_episodes": n_episodes,
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "total_steps": int(np.sum(episode_lengths)),
        "action_dim": action_dim,
        "action_mean": all_actions_arr.mean(axis=0).tolist(),
        "action_std": all_actions_arr.std(axis=0).tolist(),
    }
    if episode_rewards:
        stats["mean_reward"] = float(np.mean(episode_rewards))
        stats["std_reward"] = float(np.std(episode_rewards))

    # Layout: 2 rows — top row has episode length + reward, bottom row has action distributions
    has_rewards = len(episode_rewards) > 0
    n_top = 2 if has_rewards else 1
    n_bottom = min(action_dim, 4)

    fig = plt.figure(figsize=(4 * max(n_top, n_bottom), 8))
    gs = fig.add_gridspec(2, max(n_top, n_bottom), hspace=0.35, wspace=0.3)

    # Episode lengths
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(episode_lengths, bins=20, color="#2196F3", edgecolor="white", alpha=0.85)
    ax0.set_title("Episode Lengths", fontsize=11)
    ax0.set_xlabel("Steps")
    ax0.set_ylabel("Count")
    ax0.axvline(np.mean(episode_lengths), color="red", linestyle="--", label=f"Mean: {np.mean(episode_lengths):.1f}")
    ax0.legend(fontsize=8)

    # Reward distribution
    if has_rewards:
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.hist(episode_rewards, bins=20, color="#4CAF50", edgecolor="white", alpha=0.85)
        ax1.set_title("Episode Rewards", fontsize=11)
        ax1.set_xlabel("Total Reward")
        ax1.set_ylabel("Count")
        ax1.axvline(np.mean(episode_rewards), color="red", linestyle="--", label=f"Mean: {np.mean(episode_rewards):.1f}")
        ax1.legend(fontsize=8)

    # Action distributions
    colors = ["#FF9800", "#9C27B0", "#00BCD4", "#F44336"]
    for i in range(n_bottom):
        ax = fig.add_subplot(gs[1, i])
        ax.hist(all_actions_arr[:, i], bins=30, color=colors[i % len(colors)], edgecolor="white", alpha=0.85)
        ax.set_title(f"Action dim {i}", fontsize=11)
        ax.set_xlabel("Value")
        if i == 0:
            ax.set_ylabel("Count")

    fig.suptitle(f"Dataset: {Path(data_path).name}  ({n_episodes} episodes, {stats['total_steps']} steps)",
                 fontsize=13, fontweight="bold")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return stats


def plot_trajectory(
    data_path: str,
    episode_idx: int = 0,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot a single trajectory from an HDF5 dataset.

    Shows observation values and actions over time for one episode.

    Args:
        data_path: Path to HDF5 dataset.
        episode_idx: Episode index to plot.
        save_path: Path to save image (PNG).
        show: Display the plot interactively.
    """
    import h5py

    plt = _import_plt()

    with h5py.File(data_path, "r") as f:
        episode_keys = sorted(f.keys())
        ep_key = episode_keys[episode_idx]
        ep = f[ep_key]

        actions = ep["actions"][:]
        T = len(actions)
        timesteps = np.arange(T)

        rewards = ep["rewards"][:] if "rewards" in ep else None

        obs_data = {}
        if "observations" in ep:
            for key in ep["observations"]:
                data = ep["observations"][key][:]
                if data.ndim <= 2:  # Skip images
                    obs_data[key] = data

    # Count panels
    n_panels = 1  # actions always shown
    if rewards is not None:
        n_panels += 1
    n_panels += len(obs_data)

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    idx = 0

    # Observations
    for key, data in obs_data.items():
        ax = axes[idx]
        if data.ndim == 1:
            ax.plot(timesteps, data, label=key)
        else:
            for d in range(data.shape[1]):
                ax.plot(timesteps, data[:, d], label=f"{key}[{d}]", alpha=0.8)
        ax.set_ylabel(key)
        ax.legend(fontsize=7, ncol=min(data.shape[-1] if data.ndim > 1 else 1, 7), loc="upper right")
        ax.grid(True, alpha=0.3)
        idx += 1

    # Actions
    ax = axes[idx]
    for d in range(actions.shape[1]):
        ax.plot(timesteps, actions[:, d], label=f"action[{d}]", alpha=0.8)
    ax.set_ylabel("Actions")
    ax.legend(fontsize=8, ncol=actions.shape[1])
    ax.grid(True, alpha=0.3)
    idx += 1

    # Rewards
    if rewards is not None:
        ax = axes[idx]
        ax.plot(timesteps, rewards, color="#4CAF50", label="reward")
        ax.fill_between(timesteps, rewards, alpha=0.2, color="#4CAF50")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        idx += 1

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"Trajectory — {ep_key} ({T} steps)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ── Training Visualization ───────────────────────────────────────────────


def plot_training_metrics(
    metrics: dict[str, list[float]],
    title: str = "Training Metrics",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot training metrics (loss, LR, etc.) over steps.

    Args:
        metrics: Dict mapping metric names to lists of values.
        title: Plot title.
        save_path: Path to save image (PNG).
        show: Display the plot interactively.
    """
    plt = _import_plt()

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

    for i, (name, values) in enumerate(metrics.items()):
        ax = axes[i]
        steps = np.arange(len(values))
        color = colors[i % len(colors)]
        ax.plot(steps, values, color=color, alpha=0.8, linewidth=1)

        # Add smoothed line if enough data points
        if len(values) > 20:
            window = max(len(values) // 20, 5)
            smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(
                steps[window - 1 :], smoothed,
                color=color, linewidth=2, label=f"{name} (smoothed)",
            )

        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        if len(values) > 20:
            ax.legend(fontsize=8)

    axes[-1].set_xlabel("Step")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_rl_training(
    history: dict[str, list[float]],
    title: str = "RL Training Progress",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot RL training history (rewards, losses, episode lengths).

    Args:
        history: Dict from PPOAgent.train() with keys like
                 'episode_reward', 'episode_length', 'policy_loss', 'value_loss'.
        title: Plot title.
        save_path: Path to save image (PNG).
        show: Display the plot interactively.
    """
    plt = _import_plt()

    panel_config = [
        ("episode_reward", "Episode Reward", "#4CAF50"),
        ("episode_length", "Episode Length", "#2196F3"),
        ("policy_loss", "Policy Loss", "#F44336"),
        ("value_loss", "Value Loss", "#FF9800"),
    ]

    available = [(k, label, c) for k, label, c in panel_config if k in history and history[k]]
    n_panels = len(available)
    if n_panels == 0:
        logger.warning("No RL training data to plot.")
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for i, (key, label, color) in enumerate(available):
        ax = axes[i]
        values = history[key]
        episodes = np.arange(len(values))

        ax.plot(episodes, values, color=color, alpha=0.4, linewidth=0.8)

        # Rolling average
        if len(values) > 10:
            window = max(len(values) // 10, 5)
            smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(episodes[window - 1 :], smoothed, color=color, linewidth=2)

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Update" if "loss" in available[-1][0] else "Episode")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ── Live Simulation ──────────────────────────────────────────────────────


def run_interactive(
    env_name: str = "reach",
    n_episodes: int = 3,
    max_steps: int = 200,
    seed: int = 42,
) -> None:
    """Launch the MuJoCo interactive viewer for an environment.

    Opens a 3D viewer window where you can observe the robot.
    The robot executes random actions.

    Args:
        env_name: Environment name.
        n_episodes: Number of episodes to run.
        max_steps: Max steps per episode.
        seed: Random seed.
    """
    from simscaleai.sim.factory import make_env

    env = make_env(env_name, render_mode="human")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        logger.info(f"Episode {ep + 1}/{n_episodes}")

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if terminated or truncated:
                logger.info(f"  Done at step {step + 1}: reward={reward:.3f}")
                break

    env.close()
