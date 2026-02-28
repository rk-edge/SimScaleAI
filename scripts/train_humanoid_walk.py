#!/usr/bin/env python3
"""Train a humanoid to walk using PPO with curriculum learning.

Stages:
  0 – Stand:  learn to stay upright (low forward reward)
  1 – Walk:   learn to walk forward (full reward)
  2 – Robust: walk under external perturbations

Usage:
    python -m scripts.train_humanoid_walk [--total-steps 500000] [--device mps]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from simscaleai.rl.agents.ppo import PPOAgent, PPOConfig
from simscaleai.sim.envs.humanoid_walk_env import (
    CurriculumConfig,
    HumanoidWalkEnv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT_DIR = Path("outputs/humanoid_walk")


class RunningMeanStd:
    """Online running mean and standard deviation for observation normalization."""

    def __init__(self, shape: tuple[int, ...], clip: float = 10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, x: np.ndarray) -> None:
        batch_mean = x.mean(axis=0) if x.ndim > 1 else x
        batch_var = x.var(axis=0) if x.ndim > 1 else np.zeros_like(x)
        batch_count = x.shape[0] if x.ndim > 1 else 1

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m_2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip)


def _pick_device(requested: str) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested


def train_curriculum(
    total_steps: int = 500_000,
    device_str: str = "auto",
    seed: int = 42,
) -> dict:
    """Train humanoid with PPO + curriculum."""
    device = _pick_device(device_str)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    curriculum = CurriculumConfig(stage=0)
    env = HumanoidWalkEnv(curriculum=curriculum)

    obs_sample, _ = env.reset(seed=seed)
    obs_dim = obs_sample.shape[0]  # 49
    act_dim = env.action_space.shape[0]  # 18

    # PPO hyper-params tuned for locomotion
    ppo_config = PPOConfig(
        hidden_dim=256,
        n_layers=2,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.005,
        max_grad_norm=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        lr=3e-4,
        n_steps=2048,
        n_epochs=10,
        batch_size=64,
        total_timesteps=total_steps,
        log_every=5,
    )

    agent = PPOAgent(obs_dim=obs_dim, action_dim=act_dim, config=ppo_config, device=device)

    # ── Training loop with curriculum ──────────────────────────────────
    logger.info(f"Starting PPO training | device={device} | total_steps={total_steps}")
    logger.info(f"Curriculum stages: Stand(0) → Walk(1) → Robust(2)")

    t0 = time.perf_counter()
    all_metrics: dict[str, list] = {
        "episode_reward": [],
        "episode_length": [],
        "curriculum_stage": [],
        "forward_vel": [],
    }

    total_steps_done = 0
    n_rollouts = 0
    episode_reward = 0.0
    episode_length = 0
    episode_rewards_window: list[float] = []  # last 20 episodes for curriculum gating
    forward_vels: list[float] = []

    obs, _ = env.reset(seed=seed)
    obs_tensor = agent._flatten_obs(obs)

    while total_steps_done < total_steps:
        # ── Collect rollout ──
        agent.buffer.reset()
        for _ in range(ppo_config.n_steps):
            with torch.no_grad():
                action, log_prob, value = agent.policy.get_action(obs_tensor)

            action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            agent.buffer.add(obs_tensor, action, reward, done, log_prob, value)

            episode_reward += reward
            episode_length += 1
            total_steps_done += 1

            if "forward_vel" in info:
                forward_vels.append(info["forward_vel"])

            if done:
                all_metrics["episode_reward"].append(episode_reward)
                all_metrics["episode_length"].append(episode_length)
                all_metrics["curriculum_stage"].append(curriculum.stage)
                if forward_vels:
                    all_metrics["forward_vel"].append(float(np.mean(forward_vels)))
                forward_vels.clear()

                episode_rewards_window.append(episode_reward)
                if len(episode_rewards_window) > 20:
                    episode_rewards_window.pop(0)

                episode_reward = 0.0
                episode_length = 0
                next_obs, _ = env.reset()

            obs_tensor = agent._flatten_obs(next_obs)

        # ── Compute advantages & update ──
        with torch.no_grad():
            _, last_value = agent.policy(obs_tensor)
        agent.buffer.compute_gae(last_value.item(), ppo_config.gamma, ppo_config.gae_lambda)
        update_metrics = agent._update()

        n_rollouts += 1

        # ── Curriculum advancement ──
        if episode_rewards_window:
            avg_reward = np.mean(episode_rewards_window)
            if curriculum.stage == 0 and avg_reward >= curriculum.stand_to_walk_threshold:
                curriculum.stage = 1
                env.set_curriculum_stage(1)
                logger.info(f"★ CURRICULUM → Stage 1 (Walk) | avg_reward={avg_reward:.1f}")
            elif curriculum.stage == 1 and avg_reward >= curriculum.walk_to_robust_threshold:
                curriculum.stage = 2
                env.set_curriculum_stage(2)
                logger.info(f"★ CURRICULUM → Stage 2 (Robust) | avg_reward={avg_reward:.1f}")

        # ── Logging ──
        if n_rollouts % ppo_config.log_every == 0:
            recent = all_metrics["episode_reward"][-10:]
            avg_r = np.mean(recent) if recent else 0
            avg_len = np.mean(all_metrics["episode_length"][-10:]) if all_metrics["episode_length"] else 0
            elapsed = time.perf_counter() - t0
            fps = total_steps_done / elapsed
            logger.info(
                f"Steps: {total_steps_done:>7}/{total_steps} | "
                f"Avg reward: {avg_r:>7.1f} | Avg len: {avg_len:>5.0f} | "
                f"Stage: {curriculum.stage} | "
                f"PL: {update_metrics['policy_loss']:.4f} | "
                f"FPS: {fps:.0f}"
            )

    elapsed = time.perf_counter() - t0
    logger.info(f"Training complete in {elapsed:.1f}s ({total_steps_done / elapsed:.0f} FPS)")

    # ── Save model ──
    model_path = OUT_DIR / "humanoid_ppo.pt"
    agent.save(str(model_path))
    logger.info(f"Model saved → {model_path}")

    # ── Save metrics ──
    metrics_path = OUT_DIR / "train_metrics.json"
    serializable = {k: [float(v) for v in vs] for k, vs in all_metrics.items()}
    serializable["total_steps"] = total_steps_done
    serializable["elapsed_seconds"] = elapsed
    serializable["final_stage"] = curriculum.stage
    metrics_path.write_text(json.dumps(serializable, indent=2))
    logger.info(f"Metrics saved → {metrics_path}")

    env.close()
    return serializable


def evaluate(n_episodes: int = 20, deterministic: bool = True, device_str: str = "auto") -> dict:
    """Evaluate trained humanoid policy."""
    device = _pick_device(device_str)
    model_path = OUT_DIR / "humanoid_ppo.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}")

    env = HumanoidWalkEnv(curriculum=CurriculumConfig(stage=1))
    obs_dim = 49
    act_dim = 18

    agent = PPOAgent(obs_dim=obs_dim, action_dim=act_dim, device=device)
    agent.load(str(model_path))

    rewards, lengths, distances = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total_reward = 0.0
        steps = 0
        start_x = env.data.qpos[0]

        while True:
            action = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        distance = env.data.qpos[0] - start_x
        rewards.append(total_reward)
        lengths.append(steps)
        distances.append(distance)

    env.close()

    results = {
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
    }

    logger.info(
        f"Eval ({n_episodes} eps): reward={results['mean_reward']:.1f}±{results['std_reward']:.1f} | "
        f"steps={results['mean_length']:.0f} | dist={results['mean_distance']:.2f}m"
    )

    eval_path = OUT_DIR / "eval_results.json"
    eval_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Eval results saved → {eval_path}")

    return results


def visualize_training(metrics_path: str | None = None) -> None:
    """Plot training curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(metrics_path) if metrics_path else OUT_DIR / "train_metrics.json"
    data = json.loads(path.read_text())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Humanoid Walk — PPO + Curriculum Learning", fontsize=14, fontweight="bold")

    # Episode rewards
    ax = axes[0, 0]
    rewards = data["episode_reward"]
    ax.plot(rewards, alpha=0.3, color="steelblue", linewidth=0.5)
    # Smoothed
    if len(rewards) > 10:
        window = min(50, len(rewards) // 5)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, color="navy", linewidth=2, label=f"MA-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Episode lengths
    ax = axes[0, 1]
    lengths = data["episode_length"]
    ax.plot(lengths, alpha=0.3, color="coral", linewidth=0.5)
    if len(lengths) > 10:
        window = min(50, len(lengths) // 5)
        smoothed = np.convolve(lengths, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(lengths)), smoothed, color="darkred", linewidth=2, label=f"MA-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length (survival)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Curriculum stage
    ax = axes[1, 0]
    stages = data.get("curriculum_stage", [])
    if stages:
        ax.plot(stages, color="seagreen", linewidth=1.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Stand", "Walk", "Robust"])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Stage")
    ax.set_title("Curriculum Progression")
    ax.grid(True, alpha=0.3)

    # Forward velocity
    ax = axes[1, 1]
    fwd_vels = data.get("forward_vel", [])
    if fwd_vels:
        ax.plot(fwd_vels, alpha=0.3, color="mediumpurple", linewidth=0.5)
        if len(fwd_vels) > 10:
            window = min(50, len(fwd_vels) // 5)
            smoothed = np.convolve(fwd_vels, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(fwd_vels)), smoothed, color="indigo", linewidth=2, label=f"MA-{window}")
        ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Forward Vel (m/s)")
    ax.set_title("Average Forward Velocity")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Training curves saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train humanoid walking with PPO + curriculum")
    parser.add_argument("--total-steps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/mps/cuda/cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true", help="Skip training, just evaluate")
    args = parser.parse_args()

    if not args.eval_only:
        train_curriculum(total_steps=args.total_steps, device_str=args.device, seed=args.seed)
        visualize_training()

    evaluate(device_str=args.device)
