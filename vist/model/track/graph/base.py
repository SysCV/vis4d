"""Tracking base class."""

import abc

import torch
from pydantic import BaseModel, Field

from vist.common.registry import RegistryHolder
from vist.struct import Boxes2D


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
        self.tracks = {}  # type: ignore

    @property
    def empty(self) -> bool:
        """Whether track memory is empty."""
        return not self.tracks

    @abc.abstractmethod
    def forward_train(  # type: ignore
        self,
            inputs: List[InputSample],
            predictions: List[Instances],
    ) -> Instances:
        """Process inputs, match detections with existing tracks."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(  # type: ignore
        self,
            inputs: InputSample,
            predictions: Instances,
    ) -> Instances:
        """Process inputs, match detections with existing tracks."""
        raise NotImplementedError


def build_track_graph(cfg: TrackGraphConfig) -> BaseTrackGraph:
    """Build a tracking graph optimize from config."""
    registry = RegistryHolder.get_registry(BaseTrackGraph)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseTrackGraph)
        return module
    raise NotImplementedError(f"TrackGraph {cfg.type} not found.")
