"""Base robot environment with MuJoCo backend.

Provides the core simulation loop: load scene, step physics, render observations.
All task-specific environments inherit from this.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


@dataclass
class CameraConfig:
    """Camera rendering configuration."""

    name: str = "wrist_cam"
    width: int = 128
    height: int = 128
    fov: float = 60.0
    render_depth: bool = True
    render_segmentation: bool = False


@dataclass
class SimConfig:
    """Simulation configuration."""

    xml_path: str = ""
    dt: float = 0.002  # 500Hz physics
    control_dt: float = 0.05  # 20Hz control (25 physics steps per control step)
    max_episode_steps: int = 200
    cameras: list[CameraConfig] = field(default_factory=lambda: [CameraConfig()])
    domain_randomization: bool = False
    seed: int | None = None


class BaseRobotEnv(gym.Env, abc.ABC):
    """Abstract base environment for MuJoCo robot simulation.

    Provides:
    - MuJoCo model loading and physics stepping
    - Camera rendering (RGB, depth, segmentation)
    - Domain randomization hooks
    - Clean observation/action space interface
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config: SimConfig, render_mode: str | None = None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(config.xml_path)
        self.data = mujoco.MjData(self.model)

        # Set physics timestep
        self.model.opt.timestep = config.dt
        self.n_substeps = int(config.control_dt / config.dt)

        # Create renderers for each camera
        self._renderers: dict[str, mujoco.Renderer] = {}
        for cam in config.cameras:
            renderer = mujoco.Renderer(self.model, height=cam.height, width=cam.width)
            self._renderers[cam.name] = renderer

        # Viewer for human rendering
        self._viewer: mujoco.viewer.Handle | None = None

        # Step counter
        self._step_count = 0

        # Define spaces (subclasses must set these)
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    @abc.abstractmethod
    def _build_observation_space(self) -> spaces.Space:
        """Define the observation space for this task."""
        ...

    @abc.abstractmethod
    def _build_action_space(self) -> spaces.Space:
        """Define the action space for this task."""
        ...

    @abc.abstractmethod
    def _get_obs(self) -> dict[str, np.ndarray]:
        """Compute current observation from simulation state."""
        ...

    @abc.abstractmethod
    def _get_reward(self) -> float:
        """Compute reward for the current state."""
        ...

    @abc.abstractmethod
    def _is_terminated(self) -> bool:
        """Check if the episode has reached a terminal state (success/failure)."""
        ...

    @abc.abstractmethod
    def _reset_task(self, rng: np.random.Generator) -> None:
        """Reset task-specific state (object positions, goals, etc.)."""
        ...

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed, options=options)
        rng = np.random.default_rng(seed or self.config.seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Apply domain randomization if enabled
        if self.config.domain_randomization:
            self._apply_domain_randomization(rng)

        # Reset task-specific state
        self._reset_task(rng)

        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one control step in simulation."""
        # Apply action and step physics
        self._apply_action(action)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self._step_count >= self.config.max_episode_steps
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _apply_action(self, action: np.ndarray) -> None:
        """Apply action to the robot actuators. Override for custom control."""
        np.clip(action, self.action_space.low, self.action_space.high, out=action)
        self.data.ctrl[:] = action

    def _get_info(self) -> dict[str, Any]:
        """Return auxiliary info dict. Override to add task-specific info."""
        return {"step": self._step_count}

    # ── Rendering ──────────────────────────────────────────────────────────

    def render_camera(
        self, camera_name: str | None = None, depth: bool = False, segmentation: bool = False
    ) -> dict[str, np.ndarray]:
        """Render a named camera and return images.

        Returns:
            Dict with keys 'rgb' (H,W,3 uint8), optionally 'depth' (H,W float32)
            and 'segmentation' (H,W int32).
        """
        cam_config = self.config.cameras[0]
        if camera_name:
            cam_config = next(c for c in self.config.cameras if c.name == camera_name)

        renderer = self._renderers[cam_config.name]
        renderer.update_scene(self.data, camera=cam_config.name)

        result: dict[str, np.ndarray] = {}
        result["rgb"] = renderer.render()

        if depth or cam_config.render_depth:
            renderer.enable_depth_rendering()
            result["depth"] = renderer.render()
            renderer.disable_depth_rendering()

        if segmentation or cam_config.render_segmentation:
            renderer.enable_segmentation_rendering()
            result["segmentation"] = renderer.render()
            renderer.disable_segmentation_rendering()

        return result

    def render(self) -> np.ndarray | None:
        """Gymnasium render method."""
        if self.render_mode == "rgb_array":
            return self.render_camera()["rgb"]
        elif self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        return None

    # ── Domain Randomization ───────────────────────────────────────────────

    def _apply_domain_randomization(self, rng: np.random.Generator) -> None:
        """Apply visual and physics domain randomization.

        Uses the configurable DomainRandomizationConfig pipeline for
        systematic randomization of visual, physics, and geometry
        parameters.  Override in subclasses for task-specific additions.
        """
        from simscaleai.sim.domain_randomization import (
            DomainRandomizationConfig,
            apply_domain_randomization,
        )

        if not hasattr(self, "_dr_config"):
            self._dr_config = DomainRandomizationConfig()
        if not hasattr(self, "_dr_nominal"):
            from simscaleai.sim.domain_randomization import _cache_nominal
            self._dr_nominal = _cache_nominal(self.model)

        apply_domain_randomization(
            self.model, self.data, rng,
            config=self._dr_config,
            nominal_state=self._dr_nominal,
        )

    # ── Cleanup ────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Clean up resources."""
        for renderer in self._renderers.values():
            renderer.close()
        if self._viewer is not None:
            self._viewer.close()
        super().close()
