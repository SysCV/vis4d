"""Tracking base class."""

import abc

import torch
from pydantic import BaseModel

from openmt.core.registry import RegistryHolder


class TrackLogicConfig(BaseModel, extra="allow"):
    type: str


class BaseTracker(torch.nn.Module, metaclass=RegistryHolder):
    """Base tracker class."""

    def reset(
        self,
    ) -> None:
        """Reset tracks."""
        self.num_tracks = 0
        self.tracks = dict()

    @property
    def empty(self):
        """Whether track buffer is empty."""
        return False if self.tracks else True

    @property
    def get_ids(self):
        """Get all ids in tracker."""
        return list(self.tracks.keys())

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> None:
        """Process inputs."""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update track memory."""
        raise NotImplementedError


def build_tracker(cfg: TrackLogicConfig) -> BaseTracker:
    """Build a Tracker from config."""
    registry = RegistryHolder.get_registry(__package__)
    if cfg.type in registry:
        return registry[cfg.type](cfg)
    else:
        raise NotImplementedError(f"TrackLogic {cfg.type} not found.")
