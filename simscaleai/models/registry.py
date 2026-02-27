"""Model registry — register and instantiate model architectures by name.

Researchers add new architectures by decorating with @register_model.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch.nn as nn

logger = logging.getLogger(__name__)

# Global registry
_MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable:
    """Decorator to register a model class.

    Usage:
        @register_model("my_model")
        class MyModel(nn.Module):
            ...
    """

    def decorator(cls: type) -> type:
        if name in _MODEL_REGISTRY:
            logger.warning(f"Overwriting registered model '{name}'")
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def create_model(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a registered model by name.

    Args:
        name: Registered model name
        **kwargs: Model constructor arguments

    Returns:
        Model instance
    """
    # Ensure built-in models are registered
    _ensure_registered()

    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    model = _MODEL_REGISTRY[name](**kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created model '{name}' with {n_params:,} parameters")
    return model


def list_models() -> list[str]:
    """Return names of all registered models."""
    _ensure_registered()
    return sorted(_MODEL_REGISTRY.keys())


class ModelRegistry:
    """Convenience wrapper around the module-level registry."""

    register = staticmethod(register_model)
    create = staticmethod(create_model)
    list = staticmethod(list_models)


def _ensure_registered() -> None:
    """Import model modules to trigger @register_model decorators."""
    if _MODEL_REGISTRY:
        return
    try:
        import simscaleai.models.bc  # noqa: F401
        import simscaleai.models.vla  # noqa: F401
    except ImportError:
        pass
