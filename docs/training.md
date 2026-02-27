# Training

SimScaleAI provides a production-grade distributed training loop built on PyTorch. It supports single-GPU, multi-GPU (DDP), CPU, and Apple Silicon (MPS) training with mixed precision, gradient clipping, learning rate scheduling, checkpointing, and experiment logging.

---

## Training Configuration

All training parameters are defined in `TrainConfig`:

```python
from simscaleai.training.trainer import TrainConfig

config = TrainConfig(
    # Model
    model_name="bc",                # Model registry name
    model_kwargs={},                # Extra kwargs passed to model constructor

    # Data
    dataset_path="data/reach.h5",   # HDF5 dataset path (empty = use dummy)
    batch_size=32,                  # Training batch size
    num_workers=4,                  # DataLoader workers

    # Optimization
    lr=1e-4,                        # Learning rate
    weight_decay=1e-5,              # AdamW weight decay
    warmup_steps=500,               # Linear warmup steps
    max_steps=100_000,              # Total training steps
    grad_clip=1.0,                  # Max gradient norm

    # Mixed Precision
    use_amp=True,                   # Enable automatic mixed precision
    amp_dtype="bfloat16",           # "bfloat16" or "float16"

    # Distributed
    distributed=False,              # Enable DDP
    backend="nccl",                 # DDP backend (nccl, gloo)

    # Checkpointing
    checkpoint_dir="checkpoints",   # Save directory
    save_every=5000,                # Checkpoint interval (steps)
    resume_from=None,               # Path to resume checkpoint

    # Logging
    log_every=100,                  # Log interval (steps)
    eval_every=1000,                # Eval interval (steps)
    use_wandb=False,                # Enable WandB logging
    wandb_project="simscaleai",     # WandB project name
    wandb_run_name=None,            # WandB run name (auto-generated if None)

    # Device
    device="auto",                  # "auto", "cuda", "mps", or "cpu"
)
```

### Device Auto-Detection

The `device="auto"` setting detects the best available device:

1. **CUDA** — if `torch.cuda.is_available()`
2. **MPS** — if `torch.backends.mps.is_available()` (Apple Silicon)
3. **CPU** — fallback

---

## Trainer

### Basic Usage

```python
from torch.utils.data import DataLoader
from simscaleai.models import ModelRegistry
from simscaleai.training.trainer import Trainer, TrainConfig
from simscaleai.training.data.dataset import DummyTrajectoryDataset

# Create model
model = ModelRegistry.create("bc", state_dim=20, action_dim=4)

# Create dataset and loader
dataset = DummyTrajectoryDataset(num_samples=1000)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train
config = TrainConfig(max_steps=5000, lr=1e-4)
trainer = Trainer(model, loader, config)
metrics = trainer.train()

print(f"Final loss: {metrics['final_loss']:.4f}")
print(f"Steps completed: {metrics['global_step']}")
```

### Training Loop Internals

Each training step:

1. **Batch fetch** — pulls next batch from DataLoader (auto-restarts iterator at epoch end)
2. **Device transfer** — moves all tensors to target device
3. **AMP autocast** — wraps forward pass in mixed precision context
4. **Forward** — calls `model(batch)` → expects `{"loss": Tensor, ...}` in output
5. **Backward** — `scaler.scale(loss).backward()`
6. **Gradient clipping** — `clip_grad_norm_(max_norm=grad_clip)`
7. **Optimizer step** — `scaler.step(optimizer)` + `scaler.update()`
8. **LR scheduler step**
9. **Logging** — metrics to WandB/TensorBoard at `log_every` intervals
10. **Evaluation** — runs eval loop at `eval_every` intervals
11. **Checkpointing** — saves at `save_every` intervals

### Learning Rate Schedule

```
LR
 │    ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
 │   ╱                          ╲
 │  ╱   Cosine Annealing         ╲
 │ ╱                              ╲
 │╱                                ╲
 ├──────────────────────────────────→ Steps
 │← warmup →│
```

- **Phase 1**: Linear warmup from 0 to `lr` over `warmup_steps`
- **Phase 2**: Cosine annealing decay from `lr` to 0 over remaining steps

Implemented via `torch.optim.lr_scheduler.SequentialLR` combining `LinearLR` and `CosineAnnealingLR`.

---

## Mixed Precision (AMP)

Automatic Mixed Precision reduces memory usage and speeds up training by running parts of the forward pass in lower precision:

```python
config = TrainConfig(
    use_amp=True,
    amp_dtype="bfloat16",  # Preferred on Ampere+ GPUs and Apple Silicon
)
```

| dtype | Precision | Best for |
|-------|-----------|----------|
| `bfloat16` | Brain floating point | A100/H100, Apple M-series |
| `float16` | Half precision | V100, T4, older GPUs |

The trainer uses `torch.amp.GradScaler` for loss scaling (float16 only; bfloat16 doesn't need scaling) and `torch.amp.autocast` for the forward pass.

---

## Distributed Training (DDP)

For multi-GPU training with PyTorch DistributedDataParallel:

```python
config = TrainConfig(
    distributed=True,
    backend="nccl",       # NCCL for CUDA, gloo for CPU
    batch_size=256,
)
```

Launch with `torchrun`:

```bash
torchrun --nproc_per_node=4 -m simscaleai.tools.cli train \
    --model vla --max-steps 100000 --batch-size 256
```

**DDP behavior:**

- Model is wrapped in `DistributedDataParallel`
- `DistributedSampler` is used for data loading (epoch set each step)
- Checkpoints are only saved from rank 0
- Gradient synchronization happens automatically on backward pass

---

## Checkpointing

### Checkpoint Format

```python
checkpoint = {
    "step": int,                    # Current training step
    "model_state_dict": dict,       # Model weights
    "optimizer_state_dict": dict,   # Optimizer state (momentum, etc.)
    "scheduler_state_dict": dict,   # LR scheduler state
    "scaler_state_dict": dict,      # AMP scaler state
    "config": TrainConfig,          # Full training configuration
    "best_eval_loss": float,        # Best evaluation loss seen
}
```

### Saving

Checkpoints are saved automatically every `save_every` steps, plus a final checkpoint:

```
checkpoints/
├── step_5000.pt
├── step_10000.pt
└── final.pt
```

### Resuming

```python
config = TrainConfig(resume_from="checkpoints/step_5000.pt")
trainer = Trainer(model, loader, config)
trainer.train()  # Continues from step 5000
```

```bash
simscale train --model bc --dataset data/reach.h5 --resume checkpoints/step_5000.pt
```

---

## Datasets

### HDF5 Trajectory Dataset

The primary dataset format for trajectory data. Expected HDF5 structure:

```
dataset.h5
├── episode_0/
│   ├── observations/
│   │   ├── joint_pos    (T, 7)    float64
│   │   ├── ee_pos       (T, 3)    float64
│   │   └── image        (T, H, W, 3) uint8    [optional]
│   ├── actions          (T, 4)    float64
│   ├── rewards          (T,)      float64
│   └── @language        (attr)    str         [optional]
├── episode_1/
│   └── ...
└── episode_N/
```

```python
from simscaleai.training.data.dataset import TrajectoryDataset

dataset = TrajectoryDataset(
    data_path="data/reach.h5",
    seq_len=1,               # Timesteps per sample (1 = single-step)
    obs_keys=None,           # Observation keys to include (None = all)
    include_language=False,  # Include language instructions
    transform=None,          # Optional data augmentation
)

sample = dataset[0]
# sample = {
#     "observations": {"joint_pos": Tensor, "ee_pos": Tensor, ...},
#     "actions": Tensor (action_dim,),
#     "language": str,  # if include_language
# }
```

**Data processing:**

- Images are normalized from `[0, 255]` uint8 to `[0, 1]` float32
- Images are transposed from `(T, H, W, C)` to `(T, C, H, W)`
- Time dimension is squeezed when `seq_len == 1`
- Sliding window indexing: each sample is a contiguous `seq_len` window from an episode

### Sequence Loading

For temporal models (e.g., action chunking), use `seq_len > 1`:

```python
dataset = TrajectoryDataset("data/reach.h5", seq_len=10)
sample = dataset[0]
# sample["observations"]["joint_pos"].shape = (10, 7)
# sample["actions"].shape = (10, 4)
```

### Dummy Dataset

For testing and prototyping without real data:

```python
from simscaleai.training.data.dataset import DummyTrajectoryDataset

dataset = DummyTrajectoryDataset(
    num_samples=1000,
    obs_dim=20,
    action_dim=4,
    image_size=(128, 128),
    include_image=False,
    include_language=False,
    seed=42,
)
```

The dummy dataset generates random state/action pairs at initialization time and cycles through 5 hardcoded language instructions when `include_language=True`.

---

## Evaluation

The trainer supports periodic evaluation during training:

```python
eval_dataset = DummyTrajectoryDataset(num_samples=200)
eval_loader = DataLoader(eval_dataset, batch_size=32)

trainer = Trainer(
    model, train_loader, config,
    eval_loader=eval_loader,    # Evaluation dataloader
    eval_fn=my_custom_eval,     # Optional custom eval function
)
```

Evaluation runs every `eval_every` steps:

1. Model is set to `eval()` mode
2. All eval batches are processed (no gradients)
3. Mean loss is computed
4. Optional `eval_fn(model)` is called for custom metrics
5. Model is set back to `train()` mode

---

## Logging

### WandB

```python
config = TrainConfig(
    use_wandb=True,
    wandb_project="simscaleai",
    wandb_run_name="bc-reach-v1",  # auto-generated if None
)
```

Logged metrics per step: `train/loss`, `train/lr`, all model-returned metrics.
Logged metrics per eval: `eval/loss`, custom eval metrics.

### TensorBoard

TensorBoard logging is available as an alternative (via the same WandB interface or direct TensorBoard writer — configurable in training scripts).

---

## Complete Training Example

```python
import torch
from torch.utils.data import DataLoader, random_split
from simscaleai.models import ModelRegistry
from simscaleai.training.trainer import Trainer, TrainConfig
from simscaleai.training.data.dataset import TrajectoryDataset
from simscaleai.rl.evaluator import evaluate_policy, EvalConfig
from simscaleai.sim import make_env

# 1. Load data
dataset = TrajectoryDataset("data/reach.h5")
train_set, eval_set = random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
eval_loader = DataLoader(eval_set, batch_size=64)

# 2. Create model
model = ModelRegistry.create("bc",
    state_dim=20,
    action_dim=4,
    hidden_dim=256,
    n_layers=3,
)

# 3. Custom eval function (closed-loop in sim)
def sim_eval(model):
    env = make_env("reach")
    results = evaluate_policy(env, model.predict, EvalConfig(n_episodes=10))
    env.close()
    return results

# 4. Train
config = TrainConfig(
    max_steps=50_000,
    lr=1e-4,
    batch_size=64,
    warmup_steps=1000,
    use_amp=True,
    save_every=10_000,
    eval_every=5_000,
    use_wandb=True,
)

trainer = Trainer(model, train_loader, config,
                  eval_loader=eval_loader, eval_fn=sim_eval)
metrics = trainer.train()
```
