#!/usr/bin/env python3
"""Sim-to-Real Transfer Benchmark.

Systematically evaluates how policies trained under different conditions
transfer to unseen environment variations. Produces:

1. Transfer matrix: reward across (train_condition × eval_condition)
2. Per-parameter ablation: which DR parameter matters most
3. Heatmap visualization of the transfer matrix
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

# ── Benchmark Configuration ──────────────────────────────────────────────

N_EPISODES = 30
MAX_STEPS = 300
SEED = 200
OUT_DIR = Path("viz_output/transfer_benchmark")


# ── Helpers ──────────────────────────────────────────────────────────────

def flatten_obs(obs: dict) -> np.ndarray:
    parts = []
    for k in sorted(obs.keys()):
        if k == "image":
            continue
        parts.append(np.asarray(obs[k]).flatten())
    return np.concatenate(parts)


def eval_policy(policy_fn, env, n_episodes: int = N_EPISODES, reset_fn=None):
    """Run policy for n_episodes, return reward stats."""
    rewards, successes = [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=SEED + ep)
        if reset_fn:
            reset_fn()
        ep_reward = 0.0
        for step in range(MAX_STEPS):
            action = policy_fn(obs)
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        successes.append(info.get("success", False))
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
    }


# ── Eval Environment Configs ────────────────────────────────────────────

def _dr_config_none():
    """No randomization at all."""
    from simscaleai.sim.domain_randomization import DomainRandomizationConfig
    return DomainRandomizationConfig(
        randomize_lighting=False, randomize_camera=False,
        randomize_materials=False, randomize_friction=False,
        randomize_mass=False, randomize_damping=False,
        randomize_gains=False, randomize_object_size=False,
        randomize_table_pos=False, randomize_gravity=False,
        randomize_timestep=False,
    )


def _dr_config_light():
    """Light DR — small visual + physics noise."""
    from simscaleai.sim.domain_randomization import DomainRandomizationConfig
    return DomainRandomizationConfig(
        randomize_lighting=True, light_direction_range=0.5,
        randomize_camera=False,
        randomize_materials=True, color_noise=0.05,
        randomize_friction=True, friction_scale=(0.9, 1.1),
        randomize_mass=True, mass_scale=(0.8, 1.2),
        randomize_damping=False, randomize_gains=False,
        randomize_object_size=False,
        randomize_table_pos=False, randomize_gravity=False,
        randomize_timestep=False,
    )


def _dr_config_default():
    """Default DR — the config used for BC-DR training."""
    from simscaleai.sim.domain_randomization import DomainRandomizationConfig
    return DomainRandomizationConfig()  # all defaults


def _dr_config_heavy():
    """Heavy DR — extreme ranges to stress-test robustness."""
    from simscaleai.sim.domain_randomization import DomainRandomizationConfig
    return DomainRandomizationConfig(
        randomize_lighting=True, light_direction_range=2.0,
        light_diffuse_range=(0.1, 1.0),
        randomize_camera=True, camera_pos_noise=0.1,
        camera_fovy_range=(40.0, 80.0),
        randomize_materials=True, color_noise=0.3,
        randomize_friction=True, friction_scale=(0.4, 2.0),
        randomize_mass=True, mass_scale=(0.3, 3.0),
        randomize_damping=True, damping_scale=(0.5, 2.0),
        randomize_gains=True, kp_scale=(0.5, 2.0),
        randomize_object_size=True, size_scale=(0.5, 2.0),
        randomize_table_pos=False,
        randomize_gravity=True, gravity_noise=1.0,
        randomize_timestep=False,
    )


EVAL_CONFIGS = {
    "Clean": _dr_config_none,
    "Light DR": _dr_config_light,
    "Default DR": _dr_config_default,
    "Heavy DR": _dr_config_heavy,
}


# ── Per-Parameter Ablation Configs ──────────────────────────────────────

def _ablation_configs():
    """Create configs that randomize only ONE parameter at a time."""
    from simscaleai.sim.domain_randomization import DomainRandomizationConfig
    base = _dr_config_none()  # everything off

    ablations = {}

    # Friction only
    ablations["Friction"] = DomainRandomizationConfig(
        **{**base.__dict__, "randomize_friction": True, "friction_scale": (0.5, 1.5)}
    )
    # Mass only
    ablations["Mass"] = DomainRandomizationConfig(
        **{**base.__dict__, "randomize_mass": True, "mass_scale": (0.5, 2.0)}
    )
    # Damping only
    ablations["Damping"] = DomainRandomizationConfig(
        **{**base.__dict__, "randomize_damping": True, "damping_scale": (0.6, 1.5)}
    )
    # Gains only
    ablations["Gains"] = DomainRandomizationConfig(
        **{**base.__dict__, "randomize_gains": True, "kp_scale": (0.6, 1.5)}
    )
    # Object size only
    ablations["Obj Size"] = DomainRandomizationConfig(
        **{**base.__dict__, "randomize_object_size": True, "size_scale": (0.6, 1.5)}
    )
    # Lighting only
    ablations["Lighting"] = DomainRandomizationConfig(
        **{**base.__dict__, "randomize_lighting": True, "light_direction_range": 1.5}
    )
    # Gravity only
    ablations["Gravity"] = DomainRandomizationConfig(
        **{**base.__dict__, "randomize_gravity": True, "gravity_noise": 1.0}
    )

    return ablations


# ── Policy Loaders ──────────────────────────────────────────────────────

def load_policies():
    """Load all available trained policies."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policies = {}

    # Scripted
    from simscaleai.datagen.generator import _PickPlaceStateMachine
    sm = _PickPlaceStateMachine()
    policies["Scripted"] = (lambda obs, _sm=sm: _sm(obs), lambda: (setattr(sm, '_phase', 0), setattr(sm, '_phase_steps', 0)))

    # BC (clean)
    bc_path = "checkpoints/pick_place_bc/final.pt"
    if Path(bc_path).exists():
        from simscaleai.models.bc import BehaviorCloning
        ckpt = torch.load(bc_path, map_location=device, weights_only=False)
        ms = ckpt.get("model_state_dict", ckpt)
        sd = ms[[k for k in ms if "state_encoder.0.weight" in k][0]].shape[1]
        ad = ms[[k for k in ms if "action_head" in k and "weight" in k][-1]].shape[0]
        model = BehaviorCloning(state_dim=sd, action_dim=ad, use_image=False)
        model.load_state_dict(ms); model.to(device).eval()

        @torch.no_grad()
        def _bc(obs, _m=model, _d=device):
            s = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=_d).unsqueeze(0)
            return _m({"observations": {"state": s}})["predicted_actions"].squeeze(0).cpu().numpy()
        policies["BC"] = (_bc, None)

    # BC-DR
    bcdr_path = "checkpoints/pick_place_bc_dr/final.pt"
    if Path(bcdr_path).exists():
        from simscaleai.models.bc import BehaviorCloning
        ckpt = torch.load(bcdr_path, map_location=device, weights_only=False)
        ms = ckpt.get("model_state_dict", ckpt)
        sd = ms[[k for k in ms if "state_encoder.0.weight" in k][0]].shape[1]
        ad = ms[[k for k in ms if "action_head" in k and "weight" in k][-1]].shape[0]
        model = BehaviorCloning(state_dim=sd, action_dim=ad, use_image=False)
        model.load_state_dict(ms); model.to(device).eval()

        @torch.no_grad()
        def _bcdr(obs, _m=model, _d=device):
            s = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=_d).unsqueeze(0)
            return _m({"observations": {"state": s}})["predicted_actions"].squeeze(0).cpu().numpy()
        policies["BC-DR"] = (_bcdr, None)

    # PPO
    ppo_path = "checkpoints/pick_place_ppo/agent.pt"
    if Path(ppo_path).exists():
        from simscaleai.rl.agents.ppo import ActorCritic
        ckpt = torch.load(ppo_path, map_location=device, weights_only=False)
        ps = ckpt["policy_state_dict"]
        od = ps["feature_net.0.weight"].shape[1]
        ad = ps["actor_mean.weight"].shape[0]
        ac = ActorCritic(obs_dim=od, action_dim=ad)
        ac.load_state_dict(ps); ac.to(device).eval()

        @torch.no_grad()
        def _ppo(obs, _m=ac, _d=device):
            s = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=_d).unsqueeze(0)
            a, _, _ = _m.get_action(s, deterministic=True)
            return a.squeeze(0).cpu().numpy()
        policies["PPO"] = (_ppo, None)

    return policies


def make_env_with_dr_config(dr_config):
    """Create a pick-place env with a specific DR config injected."""
    from simscaleai.sim.factory import make_env
    env = make_env("pick_place", cameras=[], domain_randomization=True, max_episode_steps=MAX_STEPS)
    env._dr_config = dr_config
    return env


# ── Main Benchmark ──────────────────────────────────────────────────────

def run_transfer_matrix(policies):
    """Evaluate every policy × every eval config → transfer matrix."""
    policy_names = list(policies.keys())
    config_names = list(EVAL_CONFIGS.keys())

    # matrix[policy][config] = mean_reward
    matrix = np.zeros((len(policy_names), len(config_names)))
    matrix_std = np.zeros_like(matrix)
    matrix_success = np.zeros_like(matrix)

    total = len(policy_names) * len(config_names)
    done = 0

    for i, pname in enumerate(policy_names):
        policy_fn, reset_fn = policies[pname]
        for j, cname in enumerate(config_names):
            done += 1
            print(f"  [{done}/{total}] {pname} × {cname}...", end=" ", flush=True)
            dr_cfg = EVAL_CONFIGS[cname]()
            env = make_env_with_dr_config(dr_cfg)
            stats = eval_policy(policy_fn, env, n_episodes=N_EPISODES, reset_fn=reset_fn)
            env.close()
            matrix[i, j] = stats["mean_reward"]
            matrix_std[i, j] = stats["std_reward"]
            matrix_success[i, j] = stats["success_rate"]
            print(f"reward={stats['mean_reward']:.1f}±{stats['std_reward']:.1f}  success={stats['success_rate']*100:.0f}%")

    return policy_names, config_names, matrix, matrix_std, matrix_success


def run_ablation(policies, target_policy="Scripted"):
    """Ablation: eval one policy across single-parameter DR configs."""
    ablations = _ablation_configs()
    policy_fn, reset_fn = policies[target_policy]

    results = {}
    # Baseline: clean env
    print(f"  [baseline] {target_policy} × Clean...", end=" ", flush=True)
    env = make_env_with_dr_config(_dr_config_none())
    baseline = eval_policy(policy_fn, env, n_episodes=N_EPISODES, reset_fn=reset_fn)
    env.close()
    results["Clean (baseline)"] = baseline["mean_reward"]
    print(f"reward={baseline['mean_reward']:.1f}")

    for name, cfg in ablations.items():
        print(f"  [ablation] {target_policy} × {name}...", end=" ", flush=True)
        env = make_env_with_dr_config(cfg)
        stats = eval_policy(policy_fn, env, n_episodes=N_EPISODES, reset_fn=reset_fn)
        env.close()
        results[name] = stats["mean_reward"]
        drop = baseline["mean_reward"] - stats["mean_reward"]
        pct = (drop / max(abs(baseline["mean_reward"]), 1e-6)) * 100
        print(f"reward={stats['mean_reward']:.1f}  drop={drop:+.1f} ({pct:+.0f}%)")

    return results, baseline["mean_reward"]


# ── Visualization ───────────────────────────────────────────────────────

def plot_transfer_heatmap(policy_names, config_names, matrix, matrix_std):
    """Generate annotated heatmap of the transfer matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")

    # Annotate cells
    for i in range(len(policy_names)):
        for j in range(len(config_names)):
            val = matrix[i, j]
            std = matrix_std[i, j]
            color = "white" if abs(val) > 80 else "black"
            ax.text(j, i, f"{val:.0f}\n±{std:.0f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, fontsize=12)
    ax.set_yticks(range(len(policy_names)))
    ax.set_yticklabels(policy_names, fontsize=12)
    ax.set_xlabel("Evaluation Condition", fontsize=13, fontweight="bold")
    ax.set_ylabel("Trained Policy", fontsize=13, fontweight="bold")
    ax.set_title("Sim-to-Real Transfer Matrix\n(Mean Reward per Condition)", fontsize=14, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Mean Reward", shrink=0.8)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "transfer_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved heatmap → {OUT_DIR / 'transfer_heatmap.png'}")


def plot_ablation_bar(results, baseline, target_policy):
    """Bar chart of per-parameter reward drops."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    values = list(results.values())
    drops = [baseline - v for v in values]

    # Sort by drop magnitude (skip baseline)
    order = sorted(range(1, len(names)), key=lambda i: drops[i], reverse=True)
    order = [0] + order  # keep baseline first

    sorted_names = [names[i] for i in order]
    sorted_values = [values[i] for i in order]
    sorted_drops = [drops[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Absolute reward
    colors = ["#4CAF50" if i == 0 else "#FF5722" if d > 20 else "#FF9800" if d > 5 else "#2196F3"
              for i, d in enumerate(sorted_drops)]
    ax1.barh(range(len(sorted_names)), sorted_values, color=colors)
    ax1.set_yticks(range(len(sorted_names)))
    ax1.set_yticklabels(sorted_names, fontsize=11)
    ax1.set_xlabel("Mean Reward", fontsize=12)
    ax1.set_title("Reward Under Single-Parameter DR", fontsize=13, fontweight="bold")
    ax1.invert_yaxis()

    # Reward drop
    drop_colors = ["gray" if i == 0 else "#FF5722" if d > 20 else "#FF9800" if d > 5 else "#2196F3"
                   for i, d in enumerate(sorted_drops)]
    ax2.barh(range(len(sorted_names)), sorted_drops, color=drop_colors)
    ax2.set_yticks(range(len(sorted_names)))
    ax2.set_yticklabels(sorted_names, fontsize=11)
    ax2.set_xlabel("Reward Drop from Baseline", fontsize=12)
    ax2.set_title(f"Parameter Sensitivity ({target_policy})", fontsize=13, fontweight="bold")
    ax2.invert_yaxis()
    ax2.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "ablation_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ablation → {OUT_DIR / 'ablation_sensitivity.png'}")


def main():
    print("=" * 70)
    print("Sim-to-Real Transfer Benchmark")
    print(f"  Episodes: {N_EPISODES}  |  Max Steps: {MAX_STEPS}  |  Seed: {SEED}")
    print("=" * 70)

    # Load policies
    print("\nLoading policies...")
    policies = load_policies()
    print(f"  Loaded: {', '.join(policies.keys())}")

    # ── Part 1: Transfer Matrix ───────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Part 1: Transfer Matrix (Policy × Eval Condition)")
    print(f"{'─'*70}")
    p_names, c_names, matrix, matrix_std, matrix_success = run_transfer_matrix(policies)

    # Print summary table
    print(f"\n{'Policy':<12}", end="")
    for c in c_names:
        print(f" {c:>14}", end="")
    print()
    print("-" * (12 + 15 * len(c_names)))
    for i, p in enumerate(p_names):
        print(f"{p:<12}", end="")
        for j in range(len(c_names)):
            print(f" {matrix[i,j]:>7.1f}±{matrix_std[i,j]:<5.1f}", end="")
        print()

    plot_transfer_heatmap(p_names, c_names, matrix, matrix_std)

    # ── Part 2: Per-Parameter Ablation ────────────────────────────────
    print(f"\n{'─'*70}")
    print("Part 2: Per-Parameter Sensitivity Ablation (Scripted Policy)")
    print(f"{'─'*70}")
    ablation_results, baseline_reward = run_ablation(policies, "Scripted")
    plot_ablation_bar(ablation_results, baseline_reward, "Scripted")

    # Also ablate BC-DR if available
    if "BC-DR" in policies:
        print(f"\n{'─'*70}")
        print("Part 2b: Per-Parameter Sensitivity Ablation (BC-DR Policy)")
        print(f"{'─'*70}")
        abl_bcdr, base_bcdr = run_ablation(policies, "BC-DR")
        plot_ablation_bar(abl_bcdr, base_bcdr, "BC-DR")
        # Rename to avoid overwriting the Scripted ablation
        src = OUT_DIR / "ablation_sensitivity.png"
        dst = OUT_DIR / "ablation_sensitivity_bcdr.png"
        if src.exists():
            import shutil
            shutil.copy2(src, dst)
        # Re-plot scripted ablation back to the default name
        plot_ablation_bar(ablation_results, baseline_reward, "Scripted")

    # ── Save raw data ─────────────────────────────────────────────────
    raw = {
        "transfer_matrix": {
            "policies": p_names,
            "conditions": c_names,
            "rewards": matrix.tolist(),
            "stds": matrix_std.tolist(),
            "success_rates": matrix_success.tolist(),
        },
        "ablation_scripted": ablation_results,
    }
    if "BC-DR" in policies:
        raw["ablation_bcdr"] = abl_bcdr

    (OUT_DIR / "benchmark_results.json").write_text(json.dumps(raw, indent=2))
    print(f"\nSaved raw data → {OUT_DIR / 'benchmark_results.json'}")

    print(f"\n{'='*70}")
    print(f"All outputs in {OUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
