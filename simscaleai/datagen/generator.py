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
        cameras=[],
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

            # Reset stateful policies between episodes
            if hasattr(policy_fn, '_phase'):
                policy_fn._phase = 0
                policy_fn._phase_steps = 0

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
    from simscaleai.sim.factory import _ENV_REGISTRY
    env_class = type(env)

    if policy_type == "random":
        return lambda obs: env.action_space.sample()
    elif policy_type == "scripted":
        # Auto-detect env type for scripted policy
        from simscaleai.sim.envs.juggle_env import JuggleEnv
        from simscaleai.sim.envs.pick_place_env import PickPlaceEnv
        from simscaleai.sim.envs.cloth_fold_env import ClothFoldEnv
        if isinstance(env, JuggleEnv):
            return _scripted_juggle_policy
        if isinstance(env, PickPlaceEnv):
            return _PickPlaceStateMachine()
        if isinstance(env, ClothFoldEnv):
            return _ClothFoldStateMachine()
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


def _scripted_juggle_policy(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Scripted juggling policy — move paddle under lowest ball and toss upward.

    Strategy:
    1. Find the lowest ball that is falling
    2. Move paddle under it
    3. When ball is close, flick upward to toss it
    4. Alternate between balls to keep them all airborne
    """
    ee_pos = obs.get("ee_pos", np.zeros(3))
    ball_pos = obs.get("ball_pos", np.zeros(9)).reshape(3, 3)
    ball_vel = obs.get("ball_vel", np.zeros(9)).reshape(3, 3)

    # Find the ball closest to falling onto the paddle
    # Priority: lowest ball that is descending (negative z velocity)
    best_ball = 0
    best_score = float("inf")

    for i in range(3):
        bz = ball_pos[i, 2]
        bvz = ball_vel[i, 2]
        # Score: lower altitude + descending = higher priority
        score = bz + 0.3 * max(bvz, 0)  # penalize ascending balls
        if score < best_score:
            best_score = score
            best_ball = i

    target_ball_pos = ball_pos[best_ball]
    target_ball_vel = ball_vel[best_ball]

    action = np.zeros(4)

    # Move paddle horizontally under the target ball
    xy_error = target_ball_pos[:2] - ee_pos[:2]
    action[0] = np.clip(xy_error[0] * 5.0, -1, 1)
    action[1] = np.clip(xy_error[1] * 5.0, -1, 1)

    # Vertical: if ball is close and descending, toss it up
    z_dist = target_ball_pos[2] - ee_pos[2]
    if z_dist < 0.1 and target_ball_vel[2] < 0:
        # Toss upward!
        action[2] = 1.0
    elif z_dist > 0.3:
        # Ball is high, lower paddle to prepare for catch
        action[2] = -0.3
    else:
        # Track ball height loosely
        action[2] = np.clip(z_dist * 3.0, -0.5, 0.5)

    # Slight paddle tilt toward center to keep balls centered
    action[3] = np.clip(-xy_error[0] * 2.0, -0.5, 0.5)

    return np.clip(action, -1.0, 1.0)


class _PickPlaceStateMachine:
    """Scripted pick-and-place policy with 6 phases.

    Phase 0 — Approach: move EE above the object
    Phase 1 — Descend: lower EE onto the object
    Phase 2 — Grasp: close gripper and wait
    Phase 3 — Lift: raise object above table
    Phase 4 — Transport: move horizontally to target
    Phase 5 — Lower & release: descend to target and open gripper
    """

    def __init__(self) -> None:
        self._phase = 0
        self._phase_steps = 0

    def __call__(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        ee_pos = obs["ee_pos"]
        obj_pos = obs["object_pos"]
        target_pos = obs["target_pos"]

        action = np.zeros(4)
        APPROACH_HEIGHT = 0.12  # height above object for approach
        GRASP_STEPS = 10  # steps to hold gripper closed
        LIFT_HEIGHT = 0.55  # absolute z to lift to

        if self._phase == 0:
            # Move above the object (XY) with gripper open
            goal = obj_pos.copy()
            goal[2] += APPROACH_HEIGHT
            direction = goal - ee_pos
            dist = np.linalg.norm(direction)
            action[:3] = np.clip(direction * 5.0, -1, 1)
            action[3] = 1.0  # open gripper
            if dist < 0.06:
                self._phase = 1
                self._phase_steps = 0

        elif self._phase == 1:
            # Descend onto object
            goal = obj_pos.copy()
            goal[2] += 0.02  # slightly above centre
            direction = goal - ee_pos
            dist = np.linalg.norm(direction)
            action[:3] = np.clip(direction * 5.0, -1, 1)
            action[3] = 1.0  # keep open
            if dist < 0.06:
                self._phase = 2
                self._phase_steps = 0

        elif self._phase == 2:
            # Close gripper
            action[:3] = 0.0
            action[3] = -1.0  # close
            self._phase_steps += 1
            if self._phase_steps >= GRASP_STEPS:
                self._phase = 3
                self._phase_steps = 0

        elif self._phase == 3:
            # Lift — retract slightly toward base while going up
            goal = ee_pos.copy()
            goal[2] = LIFT_HEIGHT
            goal[0] = max(ee_pos[0] - 0.05, 0.25)  # retract toward base
            direction = goal - ee_pos
            action[:3] = np.clip(direction * 5.0, -1, 1)
            action[3] = -1.0  # keep closed
            if ee_pos[2] > LIFT_HEIGHT - 0.05:
                self._phase = 4
                self._phase_steps = 0

        elif self._phase == 4:
            # Transport to above target
            goal = target_pos.copy()
            goal[2] = LIFT_HEIGHT
            direction = goal - ee_pos
            dist_xy = np.linalg.norm(direction[:2])
            action[:3] = np.clip(direction * 5.0, -1, 1)
            action[3] = -1.0  # keep closed
            if dist_xy < 0.06:
                self._phase = 5
                self._phase_steps = 0

        elif self._phase == 5:
            # Lower to target and release
            goal = target_pos.copy()
            goal[2] += 0.02
            direction = goal - ee_pos
            dist = np.linalg.norm(direction)
            action[:3] = np.clip(direction * 5.0, -1, 1)
            if dist < 0.06:
                action[3] = 1.0  # open gripper to release
            else:
                action[3] = -1.0  # keep closed while lowering

        return np.clip(action, -1.0, 1.0)


class _ClothFoldStateMachine:
    """Scripted cloth-folding policy with 6 phases.

    Phase 0 — Approach:  Move EE above the grasp edge (left edge of cloth)
    Phase 1 — Descend:   Lower EE onto the grasp edge
    Phase 2 — Grasp:     Close gripper on cloth edge
    Phase 3 — Lift:      Lift the grasped edge above the table
    Phase 4 — Fold:      Sweep the edge over to the target (right) edge
    Phase 5 — Release:   Open gripper, cloth stays folded
    """

    def __init__(self) -> None:
        self._phase = 0
        self._phase_steps = 0

    def __call__(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        ee_pos = obs["ee_pos"]
        grasp_edge = obs["grasp_edge"]
        target_edge = obs["fold_target"]  # fixed initial target position

        action = np.zeros(4)
        APPROACH_HEIGHT = 0.10
        GRASP_STEPS = 10
        LIFT_HEIGHT = 0.44
        FOLD_HEIGHT = 0.43

        if self._phase == 0:
            # Move above the grasp edge (Y<0 side)
            goal = grasp_edge.copy()
            goal[2] += APPROACH_HEIGHT
            direction = goal - ee_pos
            dist = np.linalg.norm(direction)
            action[:3] = np.clip(direction * 5.0, -1, 1)
            action[3] = 1.0  # open
            if dist < 0.06:
                self._phase = 1
                self._phase_steps = 0

        elif self._phase == 1:
            # Descend onto grasp edge
            goal = grasp_edge.copy()
            goal[2] += 0.02
            direction = goal - ee_pos
            dist = np.linalg.norm(direction)
            action[:3] = np.clip(direction * 5.0, -1, 1)
            action[3] = 1.0  # open
            if dist < 0.06:
                self._phase = 2
                self._phase_steps = 0

        elif self._phase == 2:
            # Close gripper on cloth
            action[:3] = 0.0
            action[3] = -1.0  # close
            self._phase_steps += 1
            if self._phase_steps >= GRASP_STEPS:
                self._phase = 3
                self._phase_steps = 0

        elif self._phase == 3:
            # Lift the grasped edge straight up
            goal = ee_pos.copy()
            goal[2] = LIFT_HEIGHT
            direction = goal - ee_pos
            action[:3] = np.clip(direction * 5.0, -1, 1)
            action[3] = -1.0  # keep closed
            if ee_pos[2] > LIFT_HEIGHT - 0.02:
                self._phase = 4
                self._phase_steps = 0

        elif self._phase == 4:
            # Fold: sweep over to above the initial target edge, then lower.
            # Account for the grasp-to-EE offset so the CLOTH edge arrives
            # at the target, not just the end-effector.
            grasp_offset = grasp_edge - ee_pos  # cloth edge relative to EE
            if self._phase_steps < 100:
                # First: move horizontally so grasp edge is above target
                goal = target_edge - grasp_offset
                goal[2] = FOLD_HEIGHT
                direction = goal - ee_pos
                dist_xy = np.linalg.norm(direction[:2])
                action[:3] = np.clip(direction * 8.0, -1, 1)
                action[3] = -1.0
                if dist_xy < 0.03:
                    self._phase_steps = 100  # trigger lowering
                else:
                    self._phase_steps += 1
            else:
                # Then: lower to lay cloth down
                goal = target_edge - grasp_offset
                goal[2] = target_edge[2] + 0.01
                direction = goal - ee_pos
                action[:3] = np.clip(direction * 8.0, -1, 1)
                action[3] = -1.0
                if abs(ee_pos[2] - goal[2]) < 0.02:
                    self._phase = 5
                    self._phase_steps = 0

        elif self._phase == 5:
            # Release — open gripper
            action[:3] = 0.0
            action[3] = 1.0  # open
            self._phase_steps += 1

        return np.clip(action, -1.0, 1.0)
