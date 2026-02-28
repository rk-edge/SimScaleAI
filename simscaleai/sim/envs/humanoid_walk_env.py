"""Humanoid walking environment with curriculum learning.

A bipedal humanoid (21 actuated joints) trained via PPO to walk forward.
Reward shaping: forward velocity + alive bonus − energy penalty − fall penalty.
Curriculum: stand → walk → push‑recovery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from simscaleai.sim.base_env import BaseRobotEnv, SimConfig

logger = logging.getLogger(__name__)

_ASSET_PATH = str(Path(__file__).resolve().parent.parent / "assets" / "humanoid_walk.xml")

# --- Observation indices ---
# qpos: 7 (root free-joint) + 18 (actuated joints) = 25
# qvel: 6 (root free-joint) + 18 (actuated joints) = 24
# Observation vector layout:
#   torso_z(1) + torso_quat(4) + joint_pos(18) + torso_vel(3) + torso_angvel(3)
#   + joint_vel(18) + foot_contact(2) = 49
_OBS_DIM = 49
_ACT_DIM = 18  # one per actuated joint

# Default config for this env
_DEFAULT_DT = 0.002
_DEFAULT_CONTROL_DT = 0.02  # 50Hz control (10 substeps)
_DEFAULT_MAX_STEPS = 1000  # 20s episodes

# Reward weights
_FWD_REWARD_WEIGHT = 1.25
_ALIVE_BONUS = 5.0
_ENERGY_COST_WEIGHT = 0.01
_CTRL_COST_WEIGHT = 0.001
_FALL_PENALTY = -100.0

# Termination
_MIN_TORSO_Z = 0.5  # Fallen if below this
_MAX_TORSO_Z = 2.0  # Something went wrong
_MAX_TORSO_TILT = 1.0  # ~57 deg from vertical (cos(angle) < cos(1.0))


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration.

    Stages:
      0 – Stand  (just stay upright, tiny forward reward)
      1 – Walk   (full forward reward, no perturbation)
      2 – Robust (full reward + external pushes)
    """
    stage: int = 0
    # Thresholds to advance (avg episode reward over window)
    stand_to_walk_threshold: float = 40.0
    walk_to_robust_threshold: float = 120.0
    # Push parameters for stage 2
    push_interval: int = 100  # steps between pushes
    push_magnitude: float = 50.0  # Newtons


class HumanoidWalkEnv(BaseRobotEnv):
    """Bipedal humanoid locomotion environment.

    Observation (49-dim):
        torso_z               (1)  - height of torso
        torso_quat            (4)  - torso orientation quaternion
        joint_positions       (18) - actuated joint angles
        torso_linear_vel      (3)  - CoM velocity
        torso_angular_vel     (3)  - angular velocity (gyro)
        joint_velocities      (18) - actuated joint velocities
        foot_contacts         (2)  - binary foot-ground contact

    Action (18-dim):
        Normalised torques for each actuated joint [-1, 1].

    Reward:
        forward_vel * w_fwd + alive_bonus − energy_cost − ctrl_cost
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        config: SimConfig | None = None,
        render_mode: str | None = None,
        curriculum: CurriculumConfig | None = None,
    ):
        if config is None:
            config = SimConfig(
                xml_path=_ASSET_PATH,
                dt=_DEFAULT_DT,
                control_dt=_DEFAULT_CONTROL_DT,
                max_episode_steps=_DEFAULT_MAX_STEPS,
                cameras=[],  # No camera rendering needed for state-based PPO
            )
        if not config.xml_path:
            config.xml_path = _ASSET_PATH

        self.curriculum = curriculum or CurriculumConfig()
        self._prev_xpos: float = 0.0  # tracks forward progress

        super().__init__(config, render_mode)

        # Cache joint/body indices
        self._torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        self._left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")

        # Free-joint qpos slice: first 7 (pos xyz + quat wxyz)
        # Actuated joints qpos: indices 7..25
        # Free-joint qvel slice: first 6 (vel xyz + angvel xyz)
        # Actuated joints qvel: indices 6..24
        self._qpos_actuated = slice(7, 25)
        self._qvel_actuated = slice(6, 24)

        logger.info(
            f"HumanoidWalkEnv: obs_dim={_OBS_DIM}, act_dim={_ACT_DIM}, "
            f"curriculum_stage={self.curriculum.stage}"
        )

    # ── Gymnasium interface ─────────────────────────────────────────────

    def _build_observation_space(self) -> spaces.Space:
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(_OBS_DIM,),
            dtype=np.float64,
        )

    def _build_action_space(self) -> spaces.Space:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(_ACT_DIM,),
            dtype=np.float64,
        )

    def _get_obs(self) -> np.ndarray:
        """Return flat 49-dim state observation."""
        data = self.data

        torso_z = np.array([data.qpos[2]])
        torso_quat = data.qpos[3:7].copy()
        joint_pos = data.qpos[self._qpos_actuated].copy()

        torso_vel = data.qvel[0:3].copy()
        torso_angvel = data.qvel[3:6].copy()
        joint_vel = data.qvel[self._qvel_actuated].copy()

        # Foot contacts — check if sensor data has nonzero touch
        foot_contacts = np.array([
            float(data.sensordata[0] > 0.01),  # right_foot_touch
            float(data.sensordata[1] > 0.01),  # left_foot_touch
        ])

        obs = np.concatenate([
            torso_z,        # 1
            torso_quat,     # 4
            joint_pos,      # 18
            torso_vel,      # 3
            torso_angvel,   # 3
            joint_vel,      # 18
            foot_contacts,  # 2
        ])
        return obs

    def _get_reward(self) -> float:
        """Compute locomotion reward."""
        data = self.data

        # Forward velocity (along x-axis)
        xpos = data.qpos[0]
        forward_vel = (xpos - self._prev_xpos) / self.config.control_dt
        self._prev_xpos = xpos

        # Scale forward reward by curriculum stage
        if self.curriculum.stage == 0:
            fwd_reward = forward_vel * _FWD_REWARD_WEIGHT * 0.1  # Mostly stand
        else:
            fwd_reward = forward_vel * _FWD_REWARD_WEIGHT

        # Alive bonus (large — incentivises staying upright)
        alive = _ALIVE_BONUS

        # Energy cost (joint velocities × torques) — small weight
        ctrl = data.ctrl
        joint_vel = data.qvel[self._qvel_actuated]
        energy = np.sum(np.abs(ctrl * joint_vel))

        # Control cost (penalise large actions) — small weight
        ctrl_cost = np.sum(np.square(ctrl))

        # Height bonus: reward being closer to initial standing height (1.3m)
        torso_z = data.qpos[2]
        height_bonus = 2.0 * min(torso_z / 1.3, 1.0)  # max 2.0 when at full height

        reward = (
            fwd_reward
            + alive
            + height_bonus
            - _ENERGY_COST_WEIGHT * energy
            - _CTRL_COST_WEIGHT * ctrl_cost
        )

        return float(reward)

    def _is_terminated(self) -> bool:
        """Episode ends if humanoid falls."""
        torso_z = self.data.qpos[2]

        # Fell down or launched too high
        if torso_z < _MIN_TORSO_Z or torso_z > _MAX_TORSO_Z:
            return True

        # Torso tilted too far from vertical
        # quat = [w, x, y, z]; up-direction dot with z-axis ≈ 2(w²+z²)−1 for unit quat
        quat = self.data.qpos[3:7]
        # Extract the z-component of the body's z-axis from quaternion:
        # For a unit quaternion, the z-axis of the body frame is:
        # z_body_in_world = [2(xz+wy), 2(yz-wx), 1-2(x²+y²)]
        w, qx, qy, qz = quat
        upright = 1 - 2 * (qx * qx + qy * qy)  # cos(tilt angle from z)
        if upright < np.cos(_MAX_TORSO_TILT):
            return True

        return False

    def _reset_task(self, rng: np.random.Generator) -> None:
        """Reset humanoid to standing pose with small random perturbations."""
        # Set initial standing pose
        self.data.qpos[2] = 1.3  # torso height
        self.data.qpos[3] = 1.0  # quaternion w (upright)
        self.data.qpos[4:7] = 0.0  # quaternion xyz

        # Small random perturbation to joints for exploration
        self.data.qpos[self._qpos_actuated] = rng.uniform(-0.05, 0.05, size=18)
        self.data.qvel[:] = rng.uniform(-0.05, 0.05, size=self.model.nv)

        self._prev_xpos = self.data.qpos[0]

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one control step, optionally apply curriculum pushes."""
        # Apply perturbation in robust curriculum stage
        if (
            self.curriculum.stage >= 2
            and self._step_count > 0
            and self._step_count % self.curriculum.push_interval == 0
        ):
            self._apply_push()

        # Base step handles action clipping → ctrl, substep physics
        self._apply_action(action)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self._step_count >= self.config.max_episode_steps

        # Fall penalty — makes dying costly to prevent "die-fast" strategies
        if terminated:
            reward += _FALL_PENALTY

        info = self._get_info()
        info["forward_vel"] = (self.data.qpos[0] - self._prev_xpos) / self.config.control_dt if self._step_count > 1 else 0.0
        info["torso_z"] = float(self.data.qpos[2])
        info["curriculum_stage"] = self.curriculum.stage

        return obs, reward, terminated, truncated, info

    def _apply_push(self) -> None:
        """Apply a random horizontal force to the torso."""
        rng = np.random.default_rng()
        fx = rng.uniform(-1, 1) * self.curriculum.push_magnitude
        fy = rng.uniform(-1, 1) * self.curriculum.push_magnitude
        self.data.xfrc_applied[self._torso_body_id, :3] = [fx, fy, 0]

    def _get_info(self) -> dict[str, Any]:
        info = super()._get_info()
        info["torso_z"] = float(self.data.qpos[2])
        info["curriculum_stage"] = self.curriculum.stage
        return info

    # ── Curriculum management ─────────────────────────────────────────

    def set_curriculum_stage(self, stage: int) -> None:
        """Advance curriculum stage."""
        old = self.curriculum.stage
        self.curriculum.stage = min(stage, 2)
        if old != self.curriculum.stage:
            logger.info(f"Curriculum advanced: stage {old} → {self.curriculum.stage}")

    @staticmethod
    def default_config() -> SimConfig:
        """Return the default SimConfig for this environment."""
        return SimConfig(
            xml_path=_ASSET_PATH,
            dt=_DEFAULT_DT,
            control_dt=_DEFAULT_CONTROL_DT,
            max_episode_steps=_DEFAULT_MAX_STEPS,
            cameras=[],
        )
