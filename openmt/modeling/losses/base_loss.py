"""Base class for meta architectures."""

import abc

import torch

from openmt.config import LossConfig
from openmt.core.registry import RegistryHolder


class BaseLoss(torch.nn.Module, metaclass=RegistryHolder):
    """Base loss class."""

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """Loss function implementation."""
        raise NotImplementedError


def build_loss(cfg: LossConfig) -> BaseLoss:
    """
    Build the whole model architecture using meta_arch templates.
    Note that it does not load any weights from ``cfg``.
    """
    model_registry = RegistryHolder.get_registry(__package__)
    if cfg.type in model_registry:
        return model_registry[cfg.type](cfg)
    else:
        raise NotImplementedError(
            f"Loss function {cfg.tracking.type} not found."
        )
