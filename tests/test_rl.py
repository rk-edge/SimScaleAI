"""Tests for RL pipeline."""

import numpy as np
import torch
import pytest


class TestPPO:
    """Test PPO agent."""

    def test_actor_critic_forward(self):
        from simscaleai.rl.agents.ppo import ActorCritic

        ac = ActorCritic(obs_dim=10, action_dim=3, hidden_dim=32, n_layers=1)
        obs = torch.randn(4, 10)
        dist, value = ac(obs)

        assert value.shape == (4,)
        action = dist.sample()
        assert action.shape == (4, 3)

    def test_get_action(self):
        from simscaleai.rl.agents.ppo import ActorCritic

        ac = ActorCritic(obs_dim=10, action_dim=3, hidden_dim=32)
        obs = torch.randn(10)
        action, log_prob, value = ac.get_action(obs)

        assert action.shape == (3,)
        assert log_prob.shape == ()
        assert value.shape == ()

    def test_rollout_buffer(self):
        from simscaleai.rl.agents.ppo import RolloutBuffer

        buffer = RolloutBuffer(n_steps=64, obs_dim=10, action_dim=3, device=torch.device("cpu"))

        for i in range(64):
            buffer.add(
                obs=torch.randn(10),
                action=torch.randn(3),
                reward=float(np.random.randn()),
                done=False,
                log_prob=torch.tensor(0.0),
                value=torch.tensor(0.0),
            )

        buffer.compute_gae(last_value=0.0, gamma=0.99, gae_lambda=0.95)

        batches = list(buffer.get_batches(batch_size=16))
        assert len(batches) == 4
        assert batches[0]["observations"].shape[0] == 16


class TestRewards:
    """Test reward functions."""

    def test_distance_reward(self):
        from simscaleai.rl.rewards.rewards import DistanceReward

        reward_fn = DistanceReward(key_a="ee_pos", key_b="target_pos")
        obs = {"ee_pos": np.array([0.0, 0.0, 0.0]), "target_pos": np.array([1.0, 0.0, 0.0])}
        reward = reward_fn.compute(obs, np.zeros(4), {})
        assert reward == pytest.approx(-1.0)

    def test_composite_reward(self):
        from simscaleai.rl.rewards.rewards import CompositeReward, DistanceReward, SuccessBonus

        reward = CompositeReward([
            (DistanceReward(), 1.0),
            (SuccessBonus(threshold=0.5, bonus=10.0), 1.0),
        ])
        obs = {"ee_pos": np.array([0.0, 0.0, 0.0]), "target_pos": np.array([0.1, 0.0, 0.0])}
        r = reward.compute(obs, np.zeros(4), {})
        assert r == pytest.approx(-0.1 + 10.0)
