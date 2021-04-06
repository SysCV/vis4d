"""Tracking base class."""

import abc
from typing import List

import torch
from pydantic import BaseModel, Field

from openmt.core.registry import RegistryHolder
from openmt.struct import Boxes2D


class TrackLogicConfig(BaseModel, extra="allow"):
    """Base config for tracking logic."""

    type: str = Field(...)


class BaseTracker(torch.nn.Module, metaclass=RegistryHolder):
    """Base tracker class."""

    def __init__(self, cfg: TrackLogicConfig):
        super().__init__()
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        """Reset tracks."""
        self.num_tracks = 0
        self.tracks = dict()

    @property
    def empty(self) -> bool:
        """Whether track memory is empty."""
        return not self.tracks

    @property
    def get_ids(self) -> List[int]:
        """Get all ids in tracker."""
        return list(self.tracks.keys())

    @abc.abstractmethod
    def forward(
        self, detections: Boxes2D, frame_id: int, *args, **kwargs
    ) -> Boxes2D:
        """Process inputs, match detections with existing tracks."""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update track memory using matched detections."""
        raise NotImplementedError


def build_tracker(cfg: TrackLogicConfig) -> BaseTracker:
    """Build a Tracker from config."""
    registry = RegistryHolder.get_registry(__package__)
    if cfg.type in registry:
        return registry[cfg.type](cfg)
    raise NotImplementedError(f"TrackLogic {cfg.type} not found.")
