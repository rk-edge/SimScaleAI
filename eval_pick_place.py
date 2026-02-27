#!/usr/bin/env python3
"""Evaluate all pick-place policies: Scripted, BC, BC-DR, PPO, VLA.

Runs each policy for N episodes and reports reward statistics and success rates.
"""
from __future__ import annotations

import sys
import numpy as np
import torch

from simscaleai.sim.factory import make_env


N_EPISODES = 50
MAX_STEPS = 300
SEED = 123


def make_eval_env(dr: bool = False):
    return make_env(
        "pick_place",
        cameras=[],
        domain_randomization=dr,
        max_episode_steps=MAX_STEPS,
    )


def flatten_obs(obs: dict) -> np.ndarray:
    """Concatenate all obs keys (sorted, skip image) into a flat vector."""
    parts = []
    for k in sorted(obs.keys()):
        if k == "image":
            continue
        parts.append(np.asarray(obs[k]).flatten())
    return np.concatenate(parts)


def eval_policy(name: str, policy_fn, env, n_episodes: int = N_EPISODES):
    """Evaluate a policy for n_episodes and return stats."""
    rewards, successes, lengths = [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=SEED + ep)
        ep_reward = 0.0
        for step in range(MAX_STEPS):
            action = policy_fn(obs)
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        successes.append(info.get("success", False))
        lengths.append(step + 1)
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "success_rate": np.mean(successes),
        "mean_length": np.mean(lengths),
    }


# ── Policy Factories ─────────────────────────────────────────────────────────

def make_scripted_policy(env):
    """Scripted pick-place state machine (same as data generator)."""
    from simscaleai.datagen.generator import _PickPlaceStateMachine
    sm = _PickPlaceStateMachine()

    def policy(obs):
        # Reset between episodes isn't needed because eval_policy calls env.reset
        return sm(obs)

    return policy, sm


def make_bc_policy(checkpoint_path: str):
    """Load BC policy from checkpoint."""
    from simscaleai.models.bc import BehaviorCloning
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Infer dims from checkpoint
    model_state = ckpt.get("model_state_dict", ckpt)
    # Find state_dim from first linear layer
    first_key = [k for k in model_state if "state_encoder.0.weight" in k][0]
    state_dim = model_state[first_key].shape[1]
    # Find action_dim from last linear layer
    last_key = [k for k in model_state if "action_head" in k and "weight" in k][-1]
    action_dim = model_state[last_key].shape[0]

    model = BehaviorCloning(state_dim=state_dim, action_dim=action_dim, use_image=False)
    model.load_state_dict(model_state)
    model.to(device).eval()

    @torch.no_grad()
    def policy(obs):
        state = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=device).unsqueeze(0)
        batch = {"observations": {"state": state}}
        out = model(batch)
        return out["predicted_actions"].squeeze(0).cpu().numpy()

    return policy


def make_ppo_policy(checkpoint_path: str):
    """Load PPO policy from checkpoint."""
    from simscaleai.rl.agents.ppo import ActorCritic
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy_state = ckpt["policy_state_dict"]
    # Infer dims
    obs_dim = policy_state["feature_net.0.weight"].shape[1]
    action_dim = policy_state["actor_mean.weight"].shape[0]

    ac = ActorCritic(obs_dim=obs_dim, action_dim=action_dim)
    ac.load_state_dict(policy_state)
    ac.to(device).eval()

    @torch.no_grad()
    def policy(obs):
        state = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=device).unsqueeze(0)
        action, _, _ = ac.get_action(state, deterministic=True)
        return action.squeeze(0).cpu().numpy()

    return policy


def make_vla_policy(checkpoint_path: str):
    """Load VLA policy from checkpoint."""
    from simscaleai.models.vla import VisionLanguageAction
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both wrapped and raw state_dict formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model_config = ckpt.get("config", {})
        state_dict = ckpt["model_state_dict"]
    else:
        # Raw state_dict saved via torch.save(model.state_dict(), ...)
        model_config = {}
        state_dict = ckpt

    # Infer state_dim from checkpoint weights
    state_dim_from_ckpt = state_dict["state_proj.0.weight"].shape[1]
    action_dim_from_ckpt = state_dict["action_head.5.weight"].shape[0]

    model = VisionLanguageAction(
        image_size=model_config.get("image_size", 64),
        patch_size=model_config.get("patch_size", 8),
        embed_dim=model_config.get("embed_dim", 128),
        num_heads=model_config.get("num_heads", 4),
        num_layers=model_config.get("num_layers", 2),
        action_dim=action_dim_from_ckpt,
        state_dim=state_dim_from_ckpt,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()

    instruction = "pick up the red cube and place it at the green target"

    @torch.no_grad()
    def policy(obs):
        state = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=device).unsqueeze(0)
        # Dummy image (64x64x3, VLA expects channel-first)
        dummy_img = torch.zeros(1, 3, 64, 64, device=device)
        batch = {
            "observations": {
                "image": dummy_img,
                "state": state,
            },
            "language": [instruction],
        }
        out = model(batch)
        return out["predicted_actions"].squeeze(0).cpu().numpy()

    return policy


def main():
    print("=" * 70)
    print("Pick-and-Place Policy Evaluation")
    print(f"  Episodes: {N_EPISODES}  |  Max Steps: {MAX_STEPS}  |  Seed: {SEED}")
    print("=" * 70)

    results = {}

    # ── 1. Scripted Policy ────────────────────────────────────────────
    print("\n[1/5] Scripted policy...")
    env = make_eval_env(dr=False)
    scripted_fn, sm = make_scripted_policy(env)

    def scripted_with_reset(obs):
        return scripted_fn(obs)

    # We need to reset the SM between episodes
    scripted_rewards, scripted_successes, scripted_lengths = [], [], []
    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=SEED + ep)
        sm._phase = 0
        sm._phase_steps = 0
        ep_reward = 0.0
        for step in range(MAX_STEPS):
            action = scripted_fn(obs)
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            if terminated or truncated:
                break
        scripted_rewards.append(ep_reward)
        scripted_successes.append(info.get("success", False))
        scripted_lengths.append(step + 1)
    results["Scripted"] = {
        "mean_reward": np.mean(scripted_rewards),
        "std_reward": np.std(scripted_rewards),
        "success_rate": np.mean(scripted_successes),
        "mean_length": np.mean(scripted_lengths),
    }
    env.close()
    print(f"  Reward: {results['Scripted']['mean_reward']:.1f} ± {results['Scripted']['std_reward']:.1f}")
    print(f"  Success: {results['Scripted']['success_rate']*100:.1f}%")

    # ── 2. BC Policy ────────────────────────────────────────────────
    print("\n[2/5] BC policy...")
    env = make_eval_env(dr=False)
    bc_fn = make_bc_policy("checkpoints/pick_place_bc/final.pt")
    results["BC"] = eval_policy("BC", bc_fn, env)
    env.close()
    print(f"  Reward: {results['BC']['mean_reward']:.1f} ± {results['BC']['std_reward']:.1f}")
    print(f"  Success: {results['BC']['success_rate']*100:.1f}%")

    # ── 3. BC-DR Policy ────────────────────────────────────────────
    print("\n[3/5] BC-DR policy (domain randomized)...")
    env_dr = make_eval_env(dr=True)
    bc_dr_fn = make_bc_policy("checkpoints/pick_place_bc_dr/final.pt")
    results["BC-DR"] = eval_policy("BC-DR", bc_dr_fn, env_dr)
    env_dr.close()
    print(f"  Reward: {results['BC-DR']['mean_reward']:.1f} ± {results['BC-DR']['std_reward']:.1f}")
    print(f"  Success: {results['BC-DR']['success_rate']*100:.1f}%")

    # ── 4. PPO Policy ──────────────────────────────────────────────
    print("\n[4/5] PPO policy...")
    env = make_eval_env(dr=False)
    ppo_fn = make_ppo_policy("checkpoints/pick_place_ppo/agent.pt")
    results["PPO"] = eval_policy("PPO", ppo_fn, env)
    env.close()
    print(f"  Reward: {results['PPO']['mean_reward']:.1f} ± {results['PPO']['std_reward']:.1f}")
    print(f"  Success: {results['PPO']['success_rate']*100:.1f}%")

    # ── 5. VLA Policy ──────────────────────────────────────────────
    print("\n[5/5] VLA policy (language-conditioned)...")
    env = make_eval_env(dr=False)
    vla_fn = make_vla_policy("checkpoints/pick_place_vla/model.pt")
    results["VLA"] = eval_policy("VLA", vla_fn, env)
    env.close()
    print(f"  Reward: {results['VLA']['mean_reward']:.1f} ± {results['VLA']['std_reward']:.1f}")
    print(f"  Success: {results['VLA']['success_rate']*100:.1f}%")

    # ── Summary Table ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Policy':<12} {'Reward':>14} {'Success':>10} {'Avg Len':>10}")
    print("-" * 46)
    for name, stats in results.items():
        rew = f"{stats['mean_reward']:.1f}±{stats['std_reward']:.1f}"
        suc = f"{stats['success_rate']*100:.1f}%"
        length = f"{stats['mean_length']:.0f}"
        print(f"{name:<12} {rew:>14} {suc:>10} {length:>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
