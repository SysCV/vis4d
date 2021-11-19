"""Tracking base class."""

import abc
from typing import List, Optional, Union, overload

import torch
from pydantic import BaseModel, Field

from vis4d.common.registry import RegistryHolder
from vis4d.struct import Boxes2D, InputSample, LossesType


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

    @overload
    def forward(
        self, inputs: List[InputSample], predictions: List[Instances]
    ) -> Instances:
        ...

    @overload
    def forward(
        self,
        inputs: List[InputSample],
        predictions: List[Instances],
        targets: Targets,
    ) -> LossesType:
        ...

    def forward(
        self,
        inputs: List[InputSample],
        predictions: Instances,
        targets: Optional[Instances] = None,
    ) -> Union[Instances, LossesType]:  # type: ignore
        """Forward method. Decides between train / test logic."""
        if targets is not None:
            return self.forward_train(inputs, predictions, targets)
        else:
            return self.forward_test(inputs[0], predictions)

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
