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
        assert "humanoid_walk" in envs

    def test_invalid_env(self):
        from simscaleai.sim.factory import make_env

        with pytest.raises(ValueError, match="Unknown env"):
            make_env("nonexistent_env")


class TestHumanoidWalkEnv:
    """Test the humanoid walk environment."""

    def test_env_creation(self):
        """Environment creates with correct spaces."""
        from simscaleai.sim.envs.humanoid_walk_env import HumanoidWalkEnv

        env = HumanoidWalkEnv()
        assert env.observation_space.shape == (49,)
        assert env.action_space.shape == (18,)
        env.close()

    def test_reset(self):
        """Reset returns flat observation of correct shape."""
        from simscaleai.sim.envs.humanoid_walk_env import HumanoidWalkEnv

        env = HumanoidWalkEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (49,)
        assert isinstance(info, dict)
        assert "torso_z" in info
        assert info["torso_z"] > 1.0  # standing
        env.close()

    def test_step(self):
        """Step returns correct types."""
        from simscaleai.sim.envs.humanoid_walk_env import HumanoidWalkEnv

        env = HumanoidWalkEnv()
        env.reset(seed=42)
        action = np.zeros(18)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (49,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()

    def test_zero_action_survives(self):
        """With zero actions, humanoid survives at least 20 steps."""
        from simscaleai.sim.envs.humanoid_walk_env import HumanoidWalkEnv

        env = HumanoidWalkEnv()
        env.reset(seed=42)
        for i in range(20):
            _, _, terminated, _, _ = env.step(np.zeros(18))
            assert not terminated, f"Fell at step {i+1} with zero action"
        env.close()

    def test_termination_on_fall(self):
        """Large random actions cause termination."""
        from simscaleai.sim.envs.humanoid_walk_env import HumanoidWalkEnv

        env = HumanoidWalkEnv()
        env.reset(seed=42)
        terminated = False
        for _ in range(200):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated:
                break
        assert terminated, "Humanoid should fall with random actions"
        env.close()

    def test_curriculum_stages(self):
        """Curriculum stage can be set and affects reward."""
        from simscaleai.sim.envs.humanoid_walk_env import CurriculumConfig, HumanoidWalkEnv

        # Stage 0 - stand
        env = HumanoidWalkEnv(curriculum=CurriculumConfig(stage=0))
        env.reset(seed=42)
        _, r0, _, _, _ = env.step(np.zeros(18))

        # Stage 1 - walk (different forward reward weight)
        env.set_curriculum_stage(1)
        assert env.curriculum.stage == 1
        env.reset(seed=42)
        _, r1, _, _, _ = env.step(np.zeros(18))

        # Both should be positive (alive bonus)
        assert r0 > 0
        assert r1 > 0
        env.close()

    def test_fall_penalty(self):
        """Falling incurs large negative penalty."""
        from simscaleai.sim.envs.humanoid_walk_env import HumanoidWalkEnv

        env = HumanoidWalkEnv()
        env.reset(seed=42)
        last_reward = 0
        for _ in range(200):
            _, reward, terminated, _, _ = env.step(env.action_space.sample())
            last_reward = reward
            if terminated:
                break
        assert last_reward < -50, "Fall should incur large penalty"
        env.close()
