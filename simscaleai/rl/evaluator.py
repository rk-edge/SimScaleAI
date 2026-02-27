"""Closed-loop evaluation — run a policy in simulation and compute metrics.

The ultimate integration test: model controls a virtual robot in real-time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    n_episodes: int = 20
    max_steps: int = 200
    deterministic: bool = True
    render: bool = False
    save_videos: bool = False
    video_dir: str = "eval_videos"


def evaluate_policy(
    env,
    predict_fn: Callable,
    config: EvalConfig | None = None,
) -> dict[str, float]:
    """Run closed-loop evaluation of a policy in simulation.

    This is the core evaluation loop:
        observation → model → action → env.step() → observation → ...

    Args:
        env: Gymnasium environment
        predict_fn: Callable that takes obs dict → action array
        config: Evaluation configuration

    Returns:
        Metrics dict with success rate, average reward, etc.
    """
    config = config or EvalConfig()

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    successes: list[bool] = []
    all_infos: list[dict] = []

    for ep in range(config.n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        frames = []

        for step in range(config.max_steps):
            # Policy predicts action
            action = predict_fn(obs)
            if isinstance(action, np.ndarray) is False:
                action = np.array(action)

            # Step simulation
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            # Collect frames for video
            if config.save_videos and hasattr(env, "render_camera"):
                frame = env.render_camera()
                frames.append(frame.get("rgb"))

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        successes.append(info.get("success", False))
        all_infos.append(info)

        logger.debug(
            f"Episode {ep}: reward={episode_reward:.2f}, "
            f"length={episode_length}, success={info.get('success', False)}"
        )

    # Aggregate metrics
    metrics = {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
    }

    logger.info(
        f"Evaluation ({config.n_episodes} episodes): "
        f"success={metrics['success_rate']:.1%}, "
        f"reward={metrics['mean_reward']:.2f}±{metrics['std_reward']:.2f}"
    )

    return metrics
