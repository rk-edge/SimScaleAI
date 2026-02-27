"""Synthetic data generation pipeline.

Generates trajectory datasets from simulation with domain randomization.
Records (observations, actions, rewards) as HDF5 datasets for training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def generate_dataset(
    env_name: str = "reach",
    n_episodes: int = 100,
    output_path: str = "data/dataset.h5",
    policy_type: str = "random",
    domain_randomization: bool = True,
    max_steps: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a trajectory dataset from simulation.

    Args:
        env_name: Environment to generate data from
        n_episodes: Number of episodes to collect
        output_path: HDF5 output file path
        policy_type: 'random' or 'scripted'
        domain_randomization: Enable visual/physics randomization
        max_steps: Max steps per episode
        seed: Random seed

    Returns:
        Dataset statistics dict
    """
    from simscaleai.sim.factory import make_env

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = make_env(
        env_name,
        render_mode="rgb_array",
        domain_randomization=domain_randomization,
        max_episode_steps=max_steps,
        seed=seed,
    )

    # Select data collection policy
    policy_fn = _get_policy(policy_type, env)

    rng = np.random.default_rng(seed)

    total_steps = 0
    total_rewards = []
    episode_lengths = []
    successes = 0

    with h5py.File(output_path, "w") as f:
        # Store metadata
        f.attrs["env_name"] = env_name
        f.attrs["n_episodes"] = n_episodes
        f.attrs["policy_type"] = policy_type
        f.attrs["domain_randomization"] = domain_randomization
        f.attrs["seed"] = seed

        for ep_idx in range(n_episodes):
            obs, info = env.reset(seed=int(rng.integers(0, 2**31)))

            # Collect episode data
            observations: dict[str, list] = {k: [] for k in obs.keys()}
            actions: list[np.ndarray] = []
            rewards: list[float] = []

            episode_reward = 0.0

            for step in range(max_steps):
                # Store observation
                for k, v in obs.items():
                    observations[k].append(v)

                # Get action from policy
                action = policy_fn(obs)
                actions.append(action)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                episode_reward += reward

                if terminated or truncated:
                    break

            ep_len = len(actions)
            total_steps += ep_len
            total_rewards.append(episode_reward)
            episode_lengths.append(ep_len)
            if info.get("success", False):
                successes += 1

            # Write episode to HDF5
            ep_group = f.create_group(f"episode_{ep_idx:05d}")
            obs_group = ep_group.create_group("observations")
            for k, v_list in observations.items():
                obs_group.create_dataset(k, data=np.array(v_list), compression="gzip")
            ep_group.create_dataset("actions", data=np.array(actions), compression="gzip")
            ep_group.create_dataset("rewards", data=np.array(rewards), compression="gzip")

            if (ep_idx + 1) % max(1, n_episodes // 10) == 0:
                logger.info(f"Generated episode {ep_idx + 1}/{n_episodes}")

    env.close()

    stats = {
        "total_episodes": n_episodes,
        "total_steps": total_steps,
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_reward": float(np.mean(total_rewards)),
        "success_rate": successes / n_episodes,
        "output_path": output_path,
        "file_size_mb": f"{Path(output_path).stat().st_size / 1024 / 1024:.1f}",
    }

    logger.info(f"Dataset generated: {stats}")
    return stats


def _get_policy(policy_type: str, env) -> Any:
    """Get a data collection policy function."""
    if policy_type == "random":
        return lambda obs: env.action_space.sample()
    elif policy_type == "scripted":
        return _scripted_reach_policy
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def _scripted_reach_policy(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Simple scripted policy that moves toward the target."""
    ee_pos = obs.get("ee_pos", np.zeros(3))
    target = obs.get("target_pos", np.zeros(3))

    # P-controller: move toward target
    direction = target - ee_pos
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    action = np.zeros(4)
    action[:3] = direction * 0.5  # Scale down
    action[3] = 0.0  # Gripper open
    return np.clip(action, -1.0, 1.0)
