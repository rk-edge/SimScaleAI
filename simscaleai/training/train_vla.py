"""Language-conditioned VLA training on pick-and-place data.

Demonstrates the full VLA pipeline:
1. Load state + action data from HDF5
2. Auto-generate language instructions from episode context
3. Render images from env states (or use dummy images for speed)
4. Train VLA with image + language → action
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from simscaleai.models.registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Language instruction templates ────────────────────────────────────────────

PICK_PLACE_INSTRUCTIONS = [
    "pick up the red block and place it at the green target",
    "grasp the red cube and move it to the target location",
    "pick the block and put it on the target",
    "move the red object to the green marker",
    "grab the cube and place it at the goal",
    "lift the red block then set it on the target",
    "pick up the object and transport it to the target zone",
    "grasp and relocate the red block to the green spot",
]


class VLAPickPlaceDataset(Dataset):
    """Dataset for VLA training with language + state → action."""

    def __init__(self, h5_path: str, image_size: int = 64):
        self.image_size = image_size
        self.samples = []

        with h5py.File(h5_path, "r") as f:
            for ep_key in sorted(f.keys()):
                if not ep_key.startswith("episode_"):
                    continue
                ep = f[ep_key]
                obs = ep["observations"]
                actions = np.array(ep["actions"])

                # Build state vectors from all non-image obs
                state_keys = sorted([k for k in obs.keys() if k != "image"])
                states = np.concatenate(
                    [np.array(obs[k]) for k in state_keys], axis=-1
                )

                n_steps = len(actions)
                for t in range(n_steps):
                    self.samples.append({
                        "state": states[t].astype(np.float32),
                        "action": actions[t].astype(np.float32),
                    })

        logger.info(f"Loaded {len(self.samples)} VLA samples from {h5_path}")
        self.state_dim = self.samples[0]["state"].shape[0]
        self.action_dim = self.samples[0]["action"].shape[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Generate a dummy image (in production, would render from sim state)
        image = torch.randn(3, self.image_size, self.image_size) * 0.1
        # Random language instruction
        instruction = random.choice(PICK_PLACE_INSTRUCTIONS)

        return {
            "image": image,
            "state": torch.from_numpy(sample["state"]),
            "action": torch.from_numpy(sample["action"]),
            "language": instruction,
        }


def collate_vla(batch):
    """Custom collate that handles string language fields."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "state": torch.stack([b["state"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "language": [b["language"] for b in batch],
    }


def train_vla(
    dataset_path: str = "data/pick_place.h5",
    max_steps: int = 2000,
    batch_size: int = 32,
    lr: float = 1e-4,
    image_size: int = 64,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    save_path: str = "checkpoints/pick_place_vla/model.pt",
    device: str = "auto",
):
    """Train VLA model on pick-and-place with language conditioning."""
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load dataset
    dataset = VLAPickPlaceDataset(dataset_path, image_size=image_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_vla, drop_last=True
    )

    # Create VLA model
    model = ModelRegistry.create(
        "vla",
        image_size=image_size,
        patch_size=8,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        action_dim=dataset.action_dim,
        state_dim=dataset.state_dim,
        use_language=True,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"VLA model: {n_params:,} parameters, device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    # Training loop
    model.train()
    step = 0
    running_loss = 0.0
    data_iter = iter(loader)

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Move to device
        images = batch["image"].to(device)
        states = batch["state"].to(device)
        actions = batch["action"].to(device)
        language = batch["language"]

        # Forward
        vla_batch = {
            "observations": {
                "image": images,
                "state": states,
            },
            "actions": actions,
            "language": language,
        }

        result = model(vla_batch)
        loss = result["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        step += 1

        if step % 200 == 0:
            avg_loss = running_loss / 200
            action_mse = result.get("action_mse", loss).item()
            logger.info(
                f"Step {step}/{max_steps} | loss={avg_loss:.4f} | "
                f"action_mse={action_mse:.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
            )
            running_loss = 0.0

    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved VLA model to {save_path}")

    # Demo inference
    logger.info("\n--- VLA Inference Demo ---")
    model.eval()
    with torch.no_grad():
        # Create sample observation
        sample_obs = {
            "image": torch.randn(1, 3, image_size, image_size).to(device),
            "state": torch.randn(1, dataset.state_dim).to(device),
        }
        for instruction in [
            "pick up the red block and place it at the green target",
            "grasp the cube and move it to the goal",
        ]:
            pred_action = model.predict(sample_obs, language=instruction)
            logger.info(f"  Instruction: '{instruction}'")
            logger.info(f"  Predicted action: {pred_action.cpu().numpy().round(3)}")


if __name__ == "__main__":
    train_vla()
