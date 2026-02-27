# Reinforcement Learning

SimScaleAI includes a complete RL pipeline with PPO (Proximal Policy Optimization), composable reward functions, and closed-loop policy evaluation in simulation.

---

## PPO Agent

### Overview

The PPO implementation follows the PPO-Clip variant with Generalized Advantage Estimation (GAE). The agent collects rollouts in the environment, computes advantages, and updates the policy over multiple epochs of mini-batch gradient descent.

### PPO Algorithm

1. **Collect rollout**: Run policy in environment for `n_steps` steps, storing $(o_t, a_t, r_t, d_t, \log\pi(a_t|o_t), V(o_t))$
2. **Compute GAE advantages**:

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(o_{t+1}) - V(o_t)$$

3. **Update policy** (for `n_epochs` epochs, `batch_size` mini-batches):

$$\mathcal{L}_\text{policy} = -\min\left(r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)$$

$$\mathcal{L}_\text{value} = \|V_\theta(o_t) - R_t\|^2$$

$$\mathcal{L} = \mathcal{L}_\text{policy} + c_v \mathcal{L}_\text{value} - c_e H[\pi_\theta]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|o_t)}{\pi_{\theta_\text{old}}(a_t|o_t)}$ is the probability ratio.

### Configuration

```python
from simscaleai.rl.agents.ppo import PPOConfig

config = PPOConfig(
    # Network
    hidden_dim=256,          # MLP hidden layer width
    n_layers=2,              # Number of hidden layers

    # PPO hyperparameters
    clip_epsilon=0.2,        # PPO clip range
    value_loss_coef=0.5,     # Value loss coefficient (c_v)
    entropy_coef=0.01,       # Entropy bonus coefficient (c_e)
    max_grad_norm=0.5,       # Gradient clipping norm

    # GAE
    gamma=0.99,              # Discount factor
    gae_lambda=0.95,         # GAE lambda

    # Training
    lr=3e-4,                 # Adam learning rate
    n_steps=2048,            # Steps per rollout
    n_epochs=10,             # Epochs per update
    batch_size=64,           # Mini-batch size
    total_timesteps=1_000_000,  # Total environment steps
    log_every=10,            # Log every N updates
)
```

### Usage

```python
from simscaleai.sim import make_env
from simscaleai.rl.agents.ppo import PPOAgent, PPOConfig

# Create environment
env = make_env("reach")

# Create agent
config = PPOConfig(total_timesteps=100_000, lr=3e-4)
agent = PPOAgent(obs_dim=20, action_dim=4, config=config, device="cpu")

# Train
history = agent.train(env)
# history = {
#     "episode_reward": [float, ...],
#     "episode_length": [float, ...],
#     "policy_loss": [float, ...],
#     "value_loss": [float, ...],
# }

# Save / load
agent.save("checkpoints/ppo_reach.pt")
agent.load("checkpoints/ppo_reach.pt")

# Predict actions
obs, _ = env.reset()
action = agent.predict(obs, deterministic=True)  # np.ndarray (4,)
```

### ActorCritic Network

The policy is parameterized as a Gaussian actor with learned log-standard deviation:

```
Observation ──→ Shared Feature Net (MLP + Tanh) ──┬──→ Actor Mean (Linear)
                                                    │   Actor LogStd (Parameter)
                                                    │   → Normal(mean, exp(log_std))
                                                    │
                                                    └──→ Critic Value (Linear)
                                                         → scalar V(s)
```

**Initialization:**

| Component | Init Method | Gain |
|-----------|-------------|------|
| Feature layers | Orthogonal | $\sqrt{2}$ |
| Actor mean | Orthogonal | 0.01 (small actions initially) |
| Critic head | Orthogonal | 1.0 |

```python
from simscaleai.rl.agents.ppo import ActorCritic
import torch

ac = ActorCritic(obs_dim=20, action_dim=4, hidden_dim=256, n_layers=2)

obs = torch.randn(8, 20)
dist, values = ac(obs)
# dist: Normal distribution, sample shape (8, 4)
# values: Tensor (8,)

# Get action with log probability
action, log_prob, value = ac.get_action(obs[0])

# Evaluate existing actions (for PPO update)
log_prob, value, entropy = ac.evaluate_actions(obs, actions)
```

### Observation Handling

The PPO agent automatically flattens dictionary observations by concatenating all non-image values in sorted key order:

```python
obs = {
    "joint_pos": np.array([...]),   # (7,)
    "joint_vel": np.array([...]),   # (7,)
    "ee_pos": np.array([...]),      # (3,)
    "target_pos": np.array([...]),  # (3,)
}
# Flattened to: (20,) — concatenation of ee_pos(3) + joint_pos(7) + joint_vel(7) + target_pos(3)
```

The `"image"` key is automatically skipped during flattening.

### Rollout Buffer

The `RolloutBuffer` pre-allocates tensors for efficient rollout storage:

```python
from simscaleai.rl.agents.ppo import RolloutBuffer

buffer = RolloutBuffer(n_steps=2048, obs_dim=20, action_dim=4, device="cpu")

# During rollout collection
buffer.add(obs, action, reward, done, log_prob, value)

# After rollout is complete
buffer.compute_gae(last_value, gamma=0.99, gae_lambda=0.95)

# Generate mini-batches for training
for batch in buffer.get_batches(batch_size=64):
    obs, actions, old_log_probs, returns, advantages = batch
    # PPO update step...

buffer.reset()
```

---

## Reward Functions

SimScaleAI provides a modular reward function library for composing task rewards:

### Base Class

```python
from simscaleai.rl.rewards.rewards import RewardFunction

class RewardFunction(abc.ABC):
    @abc.abstractmethod
    def compute(self, obs: dict, action: np.ndarray, info: dict) -> float:
        """Compute scalar reward."""
        ...
```

### Built-in Rewards

#### DistanceReward

Negative L2 distance between two observation keys:

$$r = -\text{scale} \cdot \|o[\text{key\_a}] - o[\text{key\_b}]\|_2$$

```python
from simscaleai.rl.rewards.rewards import DistanceReward

reward_fn = DistanceReward(
    key_a="ee_pos",        # Observation key for point A
    key_b="target_pos",    # Observation key for point B
    scale=1.0,             # Reward scaling factor
)

reward = reward_fn.compute(obs, action, info)  # float
```

#### SuccessBonus

Fixed bonus when distance falls below a threshold:

$$r = \begin{cases} \text{bonus} & \text{if } \|o[\text{key\_a}] - o[\text{key\_b}]\|_2 < \text{threshold} \\ 0 & \text{otherwise} \end{cases}$$

```python
from simscaleai.rl.rewards.rewards import SuccessBonus

reward_fn = SuccessBonus(
    key_a="ee_pos",
    key_b="target_pos",
    threshold=0.05,   # Distance threshold (meters)
    bonus=1.0,        # Bonus reward value
)
```

#### ActionPenalty

Penalizes large actions for smoother control:

$$r = -\text{scale} \cdot \|a\|_2$$

```python
from simscaleai.rl.rewards.rewards import ActionPenalty

reward_fn = ActionPenalty(scale=0.01)
```

#### CompositeReward

Weighted sum of multiple reward functions:

$$r = \sum_{i} w_i \cdot r_i$$

```python
from simscaleai.rl.rewards.rewards import (
    CompositeReward, DistanceReward, SuccessBonus, ActionPenalty
)

reward_fn = CompositeReward([
    (DistanceReward("ee_pos", "target_pos"), 1.0),
    (SuccessBonus("ee_pos", "target_pos", threshold=0.05, bonus=10.0), 1.0),
    (ActionPenalty(scale=0.01), 0.5),
])

total_reward = reward_fn.compute(obs, action, info)
```

### Custom Reward Functions

```python
from simscaleai.rl.rewards.rewards import RewardFunction

class OrientationReward(RewardFunction):
    """Reward for maintaining upright object orientation."""

    def __init__(self, key="object_quat", scale=1.0):
        self.key = key
        self.scale = scale

    def compute(self, obs, action, info):
        quat = obs[self.key]
        # Reward for being close to upright (w=1, x=y=z=0)
        upright_error = 1.0 - abs(quat[0])
        return -self.scale * upright_error
```

---

## Closed-Loop Evaluation

The evaluator runs a trained policy in the simulation environment and computes performance metrics:

### Configuration

```python
from simscaleai.rl.evaluator import EvalConfig

config = EvalConfig(
    n_episodes=20,         # Number of evaluation episodes
    max_steps=200,         # Max steps per episode
    deterministic=True,    # Use deterministic actions
    render=False,          # Render to screen
    save_videos=False,     # Save video recordings
    video_dir="eval_videos",  # Video output directory
)
```

### Usage

```python
from simscaleai.sim import make_env
from simscaleai.rl.evaluator import evaluate_policy, EvalConfig

env = make_env("reach")

# With a trained PPO agent
results = evaluate_policy(
    env,
    predict_fn=agent.predict,
    config=EvalConfig(n_episodes=50),
)

# With a trained BC/VLA model
def model_predict(obs):
    # Convert obs dict to model input
    state = torch.from_numpy(
        np.concatenate([obs[k] for k in sorted(obs) if k != "image"])
    ).float().unsqueeze(0)
    with torch.no_grad():
        action = model.predict({"state": state})
    return action.squeeze(0).numpy()

results = evaluate_policy(env, predict_fn=model_predict)

print(f"Success rate:  {results['success_rate']:.1%}")
print(f"Mean reward:   {results['mean_reward']:.2f}")
print(f"Std reward:    {results['std_reward']:.2f}")
print(f"Mean length:   {results['mean_length']:.1f}")
print(f"Min reward:    {results['min_reward']:.2f}")
print(f"Max reward:    {results['max_reward']:.2f}")
```

### Return Values

| Metric | Type | Description |
|--------|------|-------------|
| `success_rate` | `float` | Fraction of episodes where `info["success"] == True` |
| `mean_reward` | `float` | Mean total episode reward |
| `std_reward` | `float` | Standard deviation of episode rewards |
| `mean_length` | `float` | Mean episode length (steps) |
| `min_reward` | `float` | Minimum total episode reward |
| `max_reward` | `float` | Maximum total episode reward |

### Evaluation Loop

```
For each episode (n_episodes):
    obs, info ← env.reset()
    total_reward ← 0

    For each step (max_steps):
        action ← predict_fn(obs)    # deterministic or stochastic
        obs, reward, terminated, truncated, info ← env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    Record: total_reward, episode_length, success
```

---

## Complete RL Training Example

```python
from simscaleai.sim import make_env
from simscaleai.rl.agents.ppo import PPOAgent, PPOConfig
from simscaleai.rl.evaluator import evaluate_policy, EvalConfig

# 1. Create environment
env = make_env("reach")

# 2. Configure PPO
config = PPOConfig(
    hidden_dim=256,
    n_layers=2,
    clip_epsilon=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    lr=3e-4,
    n_steps=2048,
    n_epochs=10,
    batch_size=64,
    total_timesteps=500_000,
)

# 3. Train
agent = PPOAgent(obs_dim=20, action_dim=4, config=config, device="cpu")
history = agent.train(env)

# 4. Evaluate
results = evaluate_policy(
    env,
    predict_fn=lambda obs: agent.predict(obs, deterministic=True),
    config=EvalConfig(n_episodes=50),
)

print(f"Success rate: {results['success_rate']:.1%}")
print(f"Mean reward: {results['mean_reward']:.2f}")

# 5. Save
agent.save("checkpoints/ppo_reach_final.pt")
env.close()
```
