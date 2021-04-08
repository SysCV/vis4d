"""RoIHead interface for backend."""

import abc
from typing import Any

import torch
from pydantic import BaseModel, Field

from openmt.core.registry import RegistryHolder


class RoIHeadConfig(BaseModel, extra="allow"):
    """Base config for RoI Heads."""

    type: str = Field(...)


class BaseRoIHead(torch.nn.Module, metaclass=RegistryHolder):  # type: ignore
    """Base roi head class."""

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore
        """Forward method.

        Process proposals, output predictions and possibly target
        assignments.
        """
        raise NotImplementedError  # pragma: no cover


def build_roi_head(cfg: RoIHeadConfig) -> BaseRoIHead:
    """Build an RoIHead from config."""
    registry = RegistryHolder.get_registry(__package__)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseRoIHead)
        return module
    raise NotImplementedError(f"RoIHead {cfg.type} not found.")
