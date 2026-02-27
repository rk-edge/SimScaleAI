"""Reward functions for robotic manipulation tasks.

Modular reward library — researchers can compose and customize
reward signals for different tasks.
"""

from __future__ import annotations

import abc

import numpy as np


class RewardFunction(abc.ABC):
    """Base class for reward functions."""

    @abc.abstractmethod
    def compute(self, obs: dict[str, np.ndarray], action: np.ndarray, info: dict) -> float:
        """Compute reward given observation, action, and info."""
        ...


class DistanceReward(RewardFunction):
    """Negative L2 distance between two points in the observation."""

    def __init__(self, key_a: str = "ee_pos", key_b: str = "target_pos", scale: float = 1.0):
        self.key_a = key_a
        self.key_b = key_b
        self.scale = scale

    def compute(self, obs: dict[str, np.ndarray], action: np.ndarray, info: dict) -> float:
        a = obs[self.key_a]
        b = obs[self.key_b]
        return -float(np.linalg.norm(a - b)) * self.scale


class SuccessBonus(RewardFunction):
    """Bonus reward when distance falls below a threshold."""

    def __init__(
        self,
        key_a: str = "ee_pos",
        key_b: str = "target_pos",
        threshold: float = 0.05,
        bonus: float = 1.0,
    ):
        self.key_a = key_a
        self.key_b = key_b
        self.threshold = threshold
        self.bonus = bonus

    def compute(self, obs: dict[str, np.ndarray], action: np.ndarray, info: dict) -> float:
        a = obs[self.key_a]
        b = obs[self.key_b]
        dist = float(np.linalg.norm(a - b))
        return self.bonus if dist < self.threshold else 0.0


class ActionPenalty(RewardFunction):
    """Penalize large actions to encourage smooth control."""

    def __init__(self, scale: float = 0.01):
        self.scale = scale

    def compute(self, obs: dict[str, np.ndarray], action: np.ndarray, info: dict) -> float:
        return -float(np.linalg.norm(action)) * self.scale


class CompositeReward(RewardFunction):
    """Combine multiple reward functions with weights."""

    def __init__(self, rewards: list[tuple[RewardFunction, float]]):
        """
        Args:
            rewards: List of (reward_function, weight) pairs
        """
        self.rewards = rewards

    def compute(self, obs: dict[str, np.ndarray], action: np.ndarray, info: dict) -> float:
        total = 0.0
        for reward_fn, weight in self.rewards:
            total += weight * reward_fn.compute(obs, action, info)
        return total
