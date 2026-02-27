"""PPO (Proximal Policy Optimization) agent for robotic control.

Implements PPO-Clip with GAE (Generalized Advantage Estimation).
The standard RL algorithm used in robotics (OpenAI, DeepMind).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    # Architecture
    hidden_dim: int = 256
    n_layers: int = 2

    # PPO
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Training
    lr: float = 3e-4
    n_steps: int = 2048  # Steps per rollout
    n_epochs: int = 10  # PPO update epochs per rollout
    batch_size: int = 64
    total_timesteps: int = 1_000_000

    # Logging
    log_every: int = 10  # Log every N rollouts


class ActorCritic(nn.Module):
    """Actor-Critic network with shared feature extractor.

    Actor outputs a Gaussian distribution over continuous actions.
    Critic outputs a scalar value estimate.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()

        # Shared feature extractor
        layers: list[nn.Module] = [nn.Linear(obs_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        self.feature_net = nn.Sequential(*layers)

        # Actor (policy) head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic (value) head
        self.critic = nn.Linear(hidden_dim, 1)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        features = self.feature_net(obs)
        action_mean = self.actor_mean(features)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        value = self.critic(features).squeeze(-1)
        return dist, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy.

        Returns: (action, log_prob, value)
        """
        dist, value = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Returns: (log_prob, value, entropy)
        """
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy


class RolloutBuffer:
    """Store rollout data for PPO updates."""

    def __init__(self, n_steps: int, obs_dim: int, action_dim: int, device: torch.device):
        self.n_steps = n_steps
        self.device = device
        self.pos = 0

        self.observations = torch.zeros(n_steps, obs_dim, device=device)
        self.actions = torch.zeros(n_steps, action_dim, device=device)
        self.rewards = torch.zeros(n_steps, device=device)
        self.dones = torch.zeros(n_steps, device=device)
        self.log_probs = torch.zeros(n_steps, device=device)
        self.values = torch.zeros(n_steps, device=device)
        self.advantages = torch.zeros(n_steps, device=device)
        self.returns = torch.zeros(n_steps, device=device)

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        self.pos += 1

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        """Compute GAE advantages and discounted returns."""
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1].item()
                next_non_terminal = 1.0 - self.dones[t].item()

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """Yield random mini-batches."""
        indices = torch.randperm(self.n_steps, device=self.device)
        for start in range(0, self.n_steps, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield {
                "observations": self.observations[batch_idx],
                "actions": self.actions[batch_idx],
                "log_probs": self.log_probs[batch_idx],
                "values": self.values[batch_idx],
                "advantages": self.advantages[batch_idx],
                "returns": self.returns[batch_idx],
            }

    def reset(self) -> None:
        self.pos = 0


class PPOAgent:
    """PPO training agent for continuous robotic control.

    Collects rollouts from the environment, computes advantages with GAE,
    and updates the policy with PPO-Clip.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: PPOConfig | None = None,
        device: str = "cpu",
    ):
        self.config = config or PPOConfig()
        self.device = torch.device(device)

        self.policy = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=self.config.hidden_dim,
            n_layers=self.config.n_layers,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr, eps=1e-5)

        self.buffer = RolloutBuffer(
            n_steps=self.config.n_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=self.device,
        )

    def train(self, env) -> dict[str, list[float]]:
        """Run PPO training loop.

        Args:
            env: Gymnasium environment

        Returns:
            Training metrics dict
        """
        total_steps = 0
        n_rollouts = 0
        metrics: dict[str, list[float]] = {
            "episode_reward": [],
            "episode_length": [],
            "policy_loss": [],
            "value_loss": [],
        }

        obs, _ = env.reset()
        obs_tensor = self._flatten_obs(obs)
        episode_reward = 0.0
        episode_length = 0

        while total_steps < self.config.total_timesteps:
            # Collect rollout
            self.buffer.reset()
            for _ in range(self.config.n_steps):
                with torch.no_grad():
                    action, log_prob, value = self.policy.get_action(obs_tensor)

                # Step environment
                action_np = action.cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated

                self.buffer.add(obs_tensor, action, reward, done, log_prob, value)

                episode_reward += reward
                episode_length += 1
                total_steps += 1

                if done:
                    metrics["episode_reward"].append(episode_reward)
                    metrics["episode_length"].append(episode_length)
                    episode_reward = 0.0
                    episode_length = 0
                    next_obs, _ = env.reset()

                obs_tensor = self._flatten_obs(next_obs)

            # Compute advantages
            with torch.no_grad():
                _, last_value = self.policy(obs_tensor)
            self.buffer.compute_gae(
                last_value.item(), self.config.gamma, self.config.gae_lambda
            )

            # PPO update
            update_metrics = self._update()
            metrics["policy_loss"].append(update_metrics["policy_loss"])
            metrics["value_loss"].append(update_metrics["value_loss"])

            n_rollouts += 1
            if n_rollouts % self.config.log_every == 0:
                recent_rewards = metrics["episode_reward"][-10:]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                logger.info(
                    f"Steps: {total_steps}/{self.config.total_timesteps} | "
                    f"Avg reward (last 10): {avg_reward:.2f} | "
                    f"Policy loss: {update_metrics['policy_loss']:.4f}"
                )

        return metrics

    def _update(self) -> dict[str, float]:
        """PPO-Clip update over collected rollout data."""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        n_updates = 0

        # Normalize advantages
        adv = self.buffer.advantages
        self.buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch["observations"], batch["actions"]
                )

                # Policy loss (PPO-Clip)
                ratio = torch.exp(log_probs - batch["log_probs"])
                adv = batch["advantages"]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch["returns"])

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
        }

    def _flatten_obs(self, obs: dict[str, np.ndarray] | np.ndarray) -> torch.Tensor:
        """Flatten observation dict to a single vector."""
        if isinstance(obs, dict):
            # Concatenate non-image observations
            parts = []
            for k, v in sorted(obs.items()):
                if k != "image":  # Skip images for state-based PPO
                    parts.append(v.flatten())
            return torch.from_numpy(np.concatenate(parts)).float().to(self.device)
        return torch.from_numpy(obs).float().to(self.device)

    def predict(self, obs: dict | np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for inference."""
        obs_tensor = self._flatten_obs(obs)
        with torch.no_grad():
            action, _, _ = self.policy.get_action(obs_tensor, deterministic=deterministic)
        return action.cpu().numpy()

    def save(self, path: str) -> None:
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
