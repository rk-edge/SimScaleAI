"""Environment factory — create envs from config names or dicts."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from simscaleai.sim.base_env import CameraConfig, SimConfig


# Registry of available environments
_ENV_REGISTRY: dict[str, type] = {}


def register_env(name: str, cls: type) -> None:
    """Register an environment class."""
    _ENV_REGISTRY[name] = cls


def make_env(
    env_name: str,
    render_mode: str | None = None,
    **kwargs: Any,
) -> gym.Env:
    """Create an environment instance by name.

    Args:
        env_name: Registered environment name (e.g., 'reach', 'pick_place')
        render_mode: 'human' for viewer, 'rgb_array' for pixel obs
        **kwargs: Override SimConfig fields

    Returns:
        Gymnasium environment instance
    """
    # Lazy registration
    if not _ENV_REGISTRY:
        _register_defaults()

    if env_name not in _ENV_REGISTRY:
        available = ", ".join(sorted(_ENV_REGISTRY.keys()))
        raise ValueError(f"Unknown env '{env_name}'. Available: {available}")

    # Build config from kwargs
    camera_kwargs = kwargs.pop("cameras", None)
    if camera_kwargs is not None:
        cameras = [CameraConfig(**c) if isinstance(c, dict) else c for c in camera_kwargs]
    else:
        cameras = [CameraConfig()]
    config = SimConfig(cameras=cameras, **kwargs)

    return _ENV_REGISTRY[env_name](config=config, render_mode=render_mode)


def _register_defaults() -> None:
    """Register built-in environments."""
    from simscaleai.sim.envs.reach_env import ReachEnv
    from simscaleai.sim.envs.pick_place_env import PickPlaceEnv
    from simscaleai.sim.envs.juggle_env import JuggleEnv
    from simscaleai.sim.envs.cloth_fold_env import ClothFoldEnv

    register_env("reach", ReachEnv)
    register_env("pick_place", PickPlaceEnv)
    register_env("juggle", JuggleEnv)
    register_env("cloth_fold", ClothFoldEnv)


def list_envs() -> list[str]:
    """Return list of registered environment names."""
    if not _ENV_REGISTRY:
        _register_defaults()
    return sorted(_ENV_REGISTRY.keys())
