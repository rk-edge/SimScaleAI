"""Tests for model architectures."""

import torch
import pytest


class TestBehaviorCloning:
    """Test Behavior Cloning model."""

    def test_forward_state_only(self):
        """BC model forward pass with state-only input."""
        from simscaleai.models.bc import BehaviorCloning

        model = BehaviorCloning(state_dim=20, action_dim=4, hidden_dim=64, n_layers=2)
        batch = {
            "observations": {"state": torch.randn(8, 20)},
            "actions": torch.randn(8, 4),
        }
        output = model(batch)

        assert "loss" in output
        assert "predicted_actions" in output
        assert output["predicted_actions"].shape == (8, 4)
        assert output["loss"].shape == ()

    def test_forward_with_image(self):
        """BC model forward pass with image input."""
        from simscaleai.models.bc import BehaviorCloning

        model = BehaviorCloning(
            state_dim=20, action_dim=4, hidden_dim=64,
            use_image=True, image_size=64, image_embed_dim=32,
        )
        batch = {
            "observations": {
                "state": torch.randn(4, 20),
                "image": torch.randn(4, 3, 64, 64),
            },
            "actions": torch.randn(4, 4),
        }
        output = model(batch)
        assert output["predicted_actions"].shape == (4, 4)

    def test_predict(self):
        """BC inference mode."""
        from simscaleai.models.bc import BehaviorCloning

        model = BehaviorCloning(state_dim=10, action_dim=3, hidden_dim=32)
        obs = {"state": torch.randn(1, 10)}
        actions = model.predict(obs)
        assert actions.shape == (1, 3)


class TestVLA:
    """Test Vision-Language-Action model."""

    def test_forward_image_only(self):
        """VLA forward pass with image only."""
        from simscaleai.models.vla import VisionLanguageAction

        model = VisionLanguageAction(
            image_size=64, patch_size=16, embed_dim=64,
            num_heads=2, num_layers=2, action_dim=4,
            use_language=False,
        )
        batch = {
            "observations": {"image": torch.randn(4, 3, 64, 64)},
            "actions": torch.randn(4, 4),
        }
        output = model(batch)
        assert output["predicted_actions"].shape == (4, 4)
        assert "loss" in output

    def test_forward_with_language(self):
        """VLA forward pass with image + language."""
        from simscaleai.models.vla import VisionLanguageAction

        model = VisionLanguageAction(
            image_size=64, patch_size=16, embed_dim=64,
            num_heads=2, num_layers=2, action_dim=4,
            use_language=True, vocab_size=100, max_text_len=32,
        )
        batch = {
            "observations": {"image": torch.randn(2, 3, 64, 64)},
            "actions": torch.randn(2, 4),
            "language": ["pick up the red cube", "place it on the target"],
        }
        output = model(batch)
        assert output["predicted_actions"].shape == (2, 4)

    def test_action_chunking(self):
        """VLA with action horizon > 1 (predicting multiple future actions)."""
        from simscaleai.models.vla import VisionLanguageAction

        model = VisionLanguageAction(
            image_size=64, patch_size=16, embed_dim=64,
            num_heads=2, num_layers=2, action_dim=4,
            action_horizon=4, use_language=False,
        )
        batch = {
            "observations": {"image": torch.randn(2, 3, 64, 64)},
            "actions": torch.randn(2, 4, 4),  # (B, horizon, action_dim)
        }
        output = model(batch)
        assert output["predicted_actions"].shape == (2, 4, 4)


class TestModelRegistry:
    """Test model registry."""

    def test_list_models(self):
        from simscaleai.models.registry import list_models

        models = list_models()
        assert "bc" in models
        assert "vla" in models

    def test_create_model(self):
        from simscaleai.models.registry import create_model

        model = create_model("bc", state_dim=10, action_dim=3, hidden_dim=32)
        assert model is not None

    def test_invalid_model(self):
        from simscaleai.models.registry import create_model

        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent_model")
