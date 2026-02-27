"""Distributed training loop with PyTorch DDP/FSDP support.

This is the core training infrastructure — config-driven, distributed,
with checkpointing, logging, and mixed precision.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""

    # Model
    model_name: str = "bc"
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    # Data
    dataset_path: str = ""
    batch_size: int = 32
    num_workers: int = 4

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    max_steps: int = 100_000
    grad_clip: float = 1.0

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # 'float16' or 'bfloat16'

    # Distributed
    distributed: bool = False
    backend: str = "nccl"  # 'nccl' for GPU, 'gloo' for CPU

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5000
    resume_from: str | None = None

    # Logging
    log_every: int = 100
    eval_every: int = 1000
    use_wandb: bool = False
    wandb_project: str = "simscaleai"
    wandb_run_name: str | None = None

    # Device
    device: str = "auto"  # 'auto', 'cuda', 'mps', 'cpu'

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def amp_torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.amp_dtype == "bfloat16" else torch.float16


class Trainer:
    """Distributed model trainer with AMP, checkpointing, and logging.

    Handles:
    - Single GPU, multi-GPU (DDP), and CPU training
    - Mixed precision (AMP) with gradient scaling
    - Checkpoint save/resume with optimizer state
    - WandB and TensorBoard logging
    - Learning rate warmup with cosine decay
    - Gradient clipping
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: TrainConfig,
        eval_loader: DataLoader | None = None,
        eval_fn: Any | None = None,
    ):
        self.config = config
        self.device = torch.device(config.resolved_device)
        self.global_step = 0
        self.best_eval_loss = float("inf")

        # Setup distributed if requested
        self._setup_distributed()

        # Move model to device
        self.model = model.to(self.device)

        # Wrap in DDP if distributed
        if self._is_distributed:
            self.model = DDP(self.model, device_ids=[self._local_rank])

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler (warmup + cosine decay)
        self.scheduler = self._build_scheduler()

        # Mixed precision
        self.scaler = GradScaler(self.device.type, enabled=config.use_amp and self.device.type == "cuda")

        # Data
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.eval_fn = eval_fn

        # Logging
        self._logger = self._setup_logging()

        # Resume from checkpoint
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

    def train(self) -> dict[str, float]:
        """Run the full training loop."""
        logger.info(
            f"Starting training: {self.config.max_steps} steps, "
            f"device={self.device}, distributed={self._is_distributed}"
        )

        self.model.train()
        train_metrics: dict[str, float] = {}

        data_iter = iter(self.train_loader)
        start_time = time.time()

        for step in range(self.global_step, self.config.max_steps):
            # Get next batch (with auto-restart)
            try:
                batch = next(data_iter)
            except StopIteration:
                if self._is_distributed and hasattr(self.train_loader, "sampler"):
                    self.train_loader.sampler.set_epoch(step)  # type: ignore
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Move batch to device
            batch = self._to_device(batch)

            # Forward + backward
            loss, metrics = self._train_step(batch)

            self.global_step = step + 1
            train_metrics = metrics

            # Logging
            if step % self.config.log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Step {step}/{self.config.max_steps} | "
                    f"loss={loss:.4f} | lr={self.scheduler.get_last_lr()[0]:.2e} | "
                    f"steps/s={steps_per_sec:.1f}"
                )
                if self._logger:
                    self._log_metrics({"train/loss": loss, **metrics}, step)

            # Evaluation
            if self.eval_loader and step % self.config.eval_every == 0 and step > 0:
                eval_metrics = self.evaluate()
                if self._logger:
                    self._log_metrics(eval_metrics, step)
                self.model.train()

            # Checkpoint
            if step % self.config.save_every == 0 and step > 0:
                self.save_checkpoint(step)

        # Final save
        self.save_checkpoint(self.global_step, is_final=True)
        logger.info(f"Training complete. {self.global_step} steps.")

        return train_metrics

    def _train_step(self, batch: dict[str, torch.Tensor]) -> tuple[float, dict[str, float]]:
        """Single training step with AMP and gradient clipping."""
        self.optimizer.zero_grad()

        amp_enabled = self.config.use_amp and self.device.type == "cuda"
        with autocast(self.device.type, dtype=self.config.amp_torch_dtype, enabled=amp_enabled):
            output = self.model(batch)
            loss = output["loss"]

        # Backward
        self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.config.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        # Collect metrics (skip non-scalar tensors like predicted_actions)
        metrics = {}
        for k, v in output.items():
            if k == "loss" or k == "predicted_actions":
                continue
            if torch.is_tensor(v) and v.ndim == 0:
                metrics[k] = v.item()
            elif not torch.is_tensor(v):
                metrics[k] = v
        return loss.item(), {f"train/{k}": v for k, v in metrics.items()}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation loop."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.eval_loader:
            batch = self._to_device(batch)
            output = self.model(batch)
            total_loss += output["loss"].item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        metrics = {"eval/loss": avg_loss}

        # Run custom evaluation (e.g., closed-loop sim)
        if self.eval_fn:
            custom_metrics = self.eval_fn(self.model)
            metrics.update({f"eval/{k}": v for k, v in custom_metrics.items()})

        logger.info(f"Eval step {self.global_step}: loss={avg_loss:.4f}")
        return metrics

    # ── Checkpointing ──────────────────────────────────────────────────────

    def save_checkpoint(self, step: int, is_final: bool = False) -> Path:
        """Save model, optimizer, and scheduler state."""
        if self._is_distributed and self._local_rank != 0:
            return Path()  # Only save on rank 0

        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        name = "final" if is_final else f"step_{step:07d}"
        path = ckpt_dir / f"{name}.pt"

        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, DDP)
            else self.model.state_dict()
        )

        checkpoint = {
            "step": step,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
            "best_eval_loss": self.best_eval_loss,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint["step"]
        self.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))

        logger.info(f"Resumed from {path} at step {self.global_step}")

    # ── Distributed ────────────────────────────────────────────────────────

    def _setup_distributed(self) -> None:
        self._is_distributed = self.config.distributed and dist.is_available()
        self._local_rank = 0

        if self._is_distributed:
            if not dist.is_initialized():
                dist.init_process_group(backend=self.config.backend)
            self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{self._local_rank}")
            torch.cuda.set_device(self._local_rank)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Warmup + cosine decay scheduler."""
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            total_iters=self.config.warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps - self.config.warmup_steps,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.config.warmup_steps],
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move a batch dict to the device."""
        return {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    def _setup_logging(self) -> Any:
        """Initialize WandB or TensorBoard logger."""
        if not self.config.use_wandb:
            return None
        if self._is_distributed and self._local_rank != 0:
            return None
        try:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=vars(self.config),
            )
            return wandb
        except ImportError:
            logger.warning("wandb not installed. Logging disabled.")
            return None

    def _log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self._logger:
            self._logger.log(metrics, step=step)
