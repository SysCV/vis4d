"""Base class for meta architectures."""

import abc
from typing import Optional

import torch
from pydantic import BaseModel

from openmt.core.registry import RegistryHolder


class LossConfig(BaseModel, extra="allow"):
    type: str
    reduction: Optional[str] = "mean"
    loss_weight: Optional[float] = 1.0


class BaseLoss(torch.nn.Module, metaclass=RegistryHolder):
    """Base loss class."""

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """Loss function implementation."""
        raise NotImplementedError


def build_loss(cfg: LossConfig) -> BaseLoss:
    """Build the loss functions for model."""
    registry = RegistryHolder.get_registry(__package__)
    if cfg.type in registry:
        return registry[cfg.type](cfg)
    else:
        raise NotImplementedError(f"Loss function {cfg.type} not found.")
