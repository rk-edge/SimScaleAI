"""Tests for simulation environments."""

import numpy as np
import pytest


class TestReachEnv:
    """Test the reach environment."""

    def test_env_creation(self):
        """Environment can be created and has correct spaces."""
        from simscaleai.sim.factory import make_env

        env = make_env("reach")
        assert env.observation_space is not None
        assert env.action_space is not None
        assert env.action_space.shape == (4,)
        env.close()

    def test_reset(self):
        """Environment resets and returns valid observation."""
        from simscaleai.sim.factory import make_env

        env = make_env("reach")
        obs, info = env.reset(seed=42)

        assert "joint_pos" in obs
        assert "ee_pos" in obs
        assert "target_pos" in obs
        assert obs["joint_pos"].shape == (7,)
        assert obs["ee_pos"].shape == (3,)
        assert obs["target_pos"].shape == (3,)

        env.close()

    def test_step(self):
        """Environment steps with valid action and returns correct types."""
        from simscaleai.sim.factory import make_env

        env = make_env("reach")
        env.reset(seed=42)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "distance" in info

        env.close()

    def test_multiple_steps(self):
        """Environment can run for multiple steps without errors."""
        from simscaleai.sim.factory import make_env

        env = make_env("reach")
        env.reset(seed=42)

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset()

        env.close()


class TestFactory:
    """Test environment factory."""

    def test_list_envs(self):
        from simscaleai.sim.factory import list_envs

        envs = list_envs()
        assert "reach" in envs
        assert "pick_place" in envs

    def test_invalid_env(self):
        from simscaleai.sim.factory import make_env

        with pytest.raises(ValueError, match="Unknown env"):
            make_env("nonexistent_env")
