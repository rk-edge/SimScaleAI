"""Evaluate all 3 juggling policies: BC, PPO, and Scripted baseline."""
import torch
import numpy as np
from simscaleai.sim.factory import make_env
from simscaleai.models.registry import create_model


def eval_policy(env, get_action_fn, n_episodes=20):
    rewards, lengths, airborne_list = [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 100)
        ep_reward, max_airborne = 0.0, 0
        for step in range(200):
            action = get_action_fn(obs)
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            max_airborne = max(max_airborne, info["balls_airborne"])
            if term or trunc:
                break
        rewards.append(ep_reward)
        lengths.append(step + 1)
        airborne_list.append(max_airborne)
    return rewards, lengths, airborne_list


def print_results(name, rewards, lengths, airborne):
    print(f"=== {name} ===")
    print(f"  Mean reward:        {np.mean(rewards):>8.1f} +/- {np.std(rewards):.1f}")
    print(f"  Mean ep length:     {np.mean(lengths):>8.1f} +/- {np.std(lengths):.1f}")
    print(f"  Max balls airborne: {np.mean(airborne):>8.1f}")
    print(f"  Best ep reward:     {np.max(rewards):>8.1f}")
    print(f"  Worst ep reward:    {np.min(rewards):>8.1f}")
    print()


from simscaleai.sim.base_env import CameraConfig, SimConfig

def make_fast_juggle():
    """Create juggle env without camera rendering for fast eval."""
    from simscaleai.sim.envs.juggle_env import JuggleEnv
    config = SimConfig(cameras=[], control_dt=0.02)
    return JuggleEnv(config=config, render_mode=None)


# --- Scripted Baseline ---
from simscaleai.datagen.generator import _scripted_juggle_policy
env = make_fast_juggle()
sc_r, sc_l, sc_a = eval_policy(env, _scripted_juggle_policy)
env.close()
print_results("Scripted Policy (Expert Baseline)", sc_r, sc_l, sc_a)

# --- BC ---
env = make_fast_juggle()
sample_obs, _ = env.reset(seed=0)
state_dim = sum(v.size for k, v in sample_obs.items() if k != "image")
model = create_model("bc", state_dim=state_dim, action_dim=4, use_image=False)
ckpt = torch.load("checkpoints/final.pt", map_location="cpu", weights_only=False)
# Load only matching keys (checkpoint has image_encoder we don't need)
model_sd = model.state_dict()
ckpt_sd = ckpt["model_state_dict"]
compatible = {k: v for k, v in ckpt_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
model_sd.update(compatible)
model.load_state_dict(model_sd)
model.eval()

def bc_action(obs):
    parts = [torch.from_numpy(obs[k]).float() for k in sorted(obs.keys()) if k != "image"]
    state = torch.cat(parts).unsqueeze(0)
    with torch.no_grad():
        pred = model({"observations": {"state": state}})
    return np.clip(pred["predicted_actions"].squeeze(0).numpy(), -1, 1)

bc_r, bc_l, bc_a = eval_policy(env, bc_action)
env.close()
print_results("BC Model (Behavior Cloning)", bc_r, bc_l, bc_a)

# --- PPO ---
from simscaleai.rl.agents.ppo import PPOAgent
env = make_fast_juggle()
obs0, _ = env.reset(seed=0)
obs_dim = sum(v.size for k, v in obs0.items() if k != "image")
agent = PPOAgent(obs_dim=obs_dim, action_dim=4)
agent_ckpt = torch.load("checkpoints/ppo_juggle.pt", map_location="cpu", weights_only=False)
agent.policy.load_state_dict(agent_ckpt["policy_state_dict"])
agent.policy.eval()

def ppo_action(obs):
    obs_vec = np.concatenate([obs[k].flatten() for k in sorted(obs.keys()) if k != "image"])
    obs_t = torch.from_numpy(obs_vec).float().unsqueeze(0)
    with torch.no_grad():
        action_t, _, _ = agent.policy.get_action(obs_t, deterministic=True)
    return np.clip(action_t.squeeze(0).numpy(), -1, 1)

ppo_r, ppo_l, ppo_a = eval_policy(env, ppo_action)
env.close()
print_results("PPO Agent (RL)", ppo_r, ppo_l, ppo_a)
