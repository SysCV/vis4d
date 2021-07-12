"""Base class for meta architectures."""

import abc
from typing import Any, Optional

import torch
from pydantic import BaseModel, Field

from openmt.common.registry import RegistryHolder


class LossConfig(BaseModel, extra="allow"):
    """Base loss config."""

    type: str = Field(...)
    reduction: str = "mean"
    loss_weight: Optional[float] = 1.0


class BaseLoss(torch.nn.Module, metaclass=RegistryHolder):  # type: ignore
    """Base loss class."""

    @abc.abstractmethod
    def forward(  # type: ignore
        self, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Loss function implementation.

        Returns the reduced loss (scalar).
        """
        raise NotImplementedError


def build_loss(cfg: LossConfig) -> BaseLoss:
    """Build the loss functions for detect."""
    registry = RegistryHolder.get_registry(BaseLoss)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseLoss)
        return module
    raise NotImplementedError(f"Loss function {cfg.type} not found.")
