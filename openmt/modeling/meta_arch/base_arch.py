"""Base class for meta architectures."""

import abc
from typing import Dict, Tuple

import torch

from openmt.config import Config
from openmt.core.registry import RegistryHolder


class BaseMetaArch(torch.nn.Module, metaclass=RegistryHolder):
    """Base model class."""

    @abc.abstractmethod
    def forward(self, batch_inputs: Tuple[Tuple[Dict[str, torch.Tensor]]]):
        """Model forward function."""
        raise NotImplementedError


def build_model(cfg: Config) -> BaseMetaArch:
    """
    Build the whole model architecture using meta_arch templates.
    Note that it does not load any weights from ``cfg``.
    """
    model_registry = RegistryHolder.get_registry(__package__)
    if cfg.tracking.type in model_registry:
        return model_registry[cfg.tracking.type](cfg)
    else:
        raise NotImplementedError(
            f"Meta architecture {cfg.tracking.type} not found."
        )
