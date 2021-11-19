"""RoI Head interface for Vis4D."""

import abc
from typing import (
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import torch
from pydantic import BaseModel, Field

from vis4d.common.bbox.samplers import SamplingResult
from vis4d.common.registry import RegistryHolder
from vis4d.struct import Boxes2D, InputSample, LossesType, TLabelInstance


class BaseRoIHeadConfig(BaseModel, extra="allow"):
    """Base config for RoI head."""

    type: str = Field(...)


class BaseRoIHead(torch.nn.Module, Generic[TLabelInstance], metaclass=RegistryHolder):  # type: ignore
    """Base RoI head class."""

    @overload
    def forward(
        self,
        inputs: List[InputSample],
        predictions: List[Instances],
        features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Sequence[TLabelInstance]:
        ...

    @overload
    def forward(
        self,
        inputs: List[InputSample],
        boxes: List[Boxes2D],
        targets: Targets,
        features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[LossesType, Optional[SamplingResult]]:
        ...

    def forward(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        targets: Optional[Targets] = None,
        features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[
        Tuple[LossesType, Optional[SamplingResult]], Sequence[TLabelInstance]
    ]:
        if targets is not None:
            return self.forward_train(inputs, boxes, targets, features)
        return self.forward_train(inputs, boxes, features)

    @abc.abstractmethod
    def forward_train(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        targets: TLabelInstance,
        features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[LossesType, Optional[SamplingResult]]:
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            boxes: Input boxes to apply RoIHead on.

        Returns:
            LossesType: A dict of scalar loss tensors.
            Optional[List[SamplingResult]]: Sampling result.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Sequence[TLabelInstance]:
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            boxes: Input boxes to apply RoIHead on.

        Returns:
            Sequence[TLabelInstance]: Prediction output.
        """
        raise NotImplementedError


def build_roi_head(cfg: BaseRoIHeadConfig) -> BaseRoIHead[TLabelInstance]:
    """Build a roi head from config."""
    registry = RegistryHolder.get_registry(BaseRoIHead)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseRoIHead)
        return module
    raise NotImplementedError(f"RoIHead {cfg.type} not found.")
