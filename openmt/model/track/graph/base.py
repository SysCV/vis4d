"""Tracking base class."""

import abc

import torch
from pydantic import BaseModel, Field

from openmt.common.registry import RegistryHolder
from openmt.struct import Boxes2D


class TrackGraphConfig(BaseModel, extra="allow"):
    """Base config for tracking graph optimization."""

    type: str = Field(...)


class BaseTrackGraph(torch.nn.Module, metaclass=RegistryHolder):  # type: ignore # pylint: disable=line-too-long
    """Base class for tracking graph optimization."""

    def __init__(self, cfg: TrackGraphConfig):
        """Init."""
        super().__init__()
        self.cfg_base = cfg
        self.reset()

    def reset(self) -> None:
        """Reset tracks."""
        self.num_tracks = 0
        self.tracks = dict()  # type: ignore

    @property
    def empty(self) -> bool:
        """Whether track memory is empty."""
        return not self.tracks

    @abc.abstractmethod
    def forward(  # type: ignore
        self, detections: Boxes2D, frame_id: int, *args, **kwargs
    ) -> Boxes2D:
        """Process inputs, match detections with existing tracks."""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None:  # type: ignore
        """Update track memory using matched detections."""
        raise NotImplementedError


def build_track_graph(cfg: TrackGraphConfig) -> BaseTrackGraph:
    """Build a tracking graph optimizer from config."""
    registry = RegistryHolder.get_registry(BaseTrackGraph)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseTrackGraph)
        return module
    raise NotImplementedError(f"TrackGraph {cfg.type} not found.")
