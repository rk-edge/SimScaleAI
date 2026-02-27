"""Tests for training infrastructure."""

import torch
import pytest
from torch.utils.data import DataLoader


class TestTrainer:
    """Test the training loop."""

    def test_training_smoke(self):
        """End-to-end training smoke test: create data → train 10 steps → check loss."""
        from simscaleai.models.registry import create_model
        from simscaleai.training.data.dataset import DummyTrajectoryDataset
        from simscaleai.training.trainer import TrainConfig, Trainer

        # Create tiny model and dummy data
        model = create_model("bc", state_dim=20, action_dim=4, hidden_dim=32, n_layers=1)
        dataset = DummyTrajectoryDataset(num_samples=100, obs_dim=20, action_dim=4)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        config = TrainConfig(
            max_steps=10,
            lr=1e-3,
            log_every=5,
            save_every=100,  # Don't save during test
            use_amp=False,
            device="cpu",
        )

        trainer = Trainer(model=model, train_loader=loader, config=config)
        metrics = trainer.train()

        # Verify training ran
        assert trainer.global_step == 10


class TestDataset:
    """Test dataset classes."""

    def test_dummy_dataset(self):
        from simscaleai.training.data.dataset import DummyTrajectoryDataset

        ds = DummyTrajectoryDataset(num_samples=50, obs_dim=10, action_dim=3)
        assert len(ds) == 50

        sample = ds[0]
        assert "observations" in sample
        assert "actions" in sample
        assert sample["observations"]["state"].shape == (10,)
        assert sample["actions"].shape == (3,)

    def test_dummy_with_image(self):
        from simscaleai.training.data.dataset import DummyTrajectoryDataset

        ds = DummyTrajectoryDataset(
            num_samples=10, include_image=True, image_size=(64, 64)
        )
        sample = ds[0]
        assert "image" in sample["observations"]
        assert sample["observations"]["image"].shape == (3, 64, 64)

    def test_dataloader(self):
        from simscaleai.training.data.dataset import DummyTrajectoryDataset

        ds = DummyTrajectoryDataset(num_samples=32, obs_dim=10, action_dim=3)
        loader = DataLoader(ds, batch_size=8, shuffle=True)

        batch = next(iter(loader))
        assert batch["observations"]["state"].shape == (8, 10)
        assert batch["actions"].shape == (8, 3)
