"""Panoptic Head interface for Vis4D."""

import abc
from typing import List, Optional, Tuple, Union, overload

from pydantic import BaseModel, Field

from vis4d.common.module import Vis4DModule
from vis4d.common.registry import RegistryHolder
from vis4d.struct import (
    InputSample,
    InstanceMasks,
    LabelInstances,
    LossesType,
    SemanticMasks,
)

PanopticMasks = Tuple[List[InstanceMasks], List[SemanticMasks]]


class BasePanopticHeadConfig(BaseModel, extra="allow"):
    """Base config for Panoptic head."""

    type: str = Field(...)


class BasePanopticHead(Vis4DModule[LossesType, PanopticMasks]):
    """Base Panoptic head class."""

    @overload  # type: ignore[override]
    def __call__(
        self, inputs: InputSample, predictions: LabelInstances
    ) -> PanopticMasks:  # noqa: D102
        ...

    @overload
    def __call__(
        self,
        inputs: InputSample,
        predictions: LabelInstances,
        targets: LabelInstances,
    ) -> LossesType:
        ...

    def __call__(
        self,
        inputs: InputSample,
        predictions: LabelInstances,
        targets: Optional[LabelInstances] = None,
    ) -> Union[LossesType, PanopticMasks]:
        """Base Panoptic head forward.

        Args:
            inputs: Model Inputs, batched.
            features: Input feature maps.
            targets: Container with targets, e.g. Boxes2D / 3D, Masks, ...

        Returns:
            LossesType / PanopticMasks: In train mode, return losses. In test
            mode, return predictions.
        """
        if targets is not None:
            return self.forward_train(inputs, predictions, targets)
        return self.forward_test(inputs, predictions)

    @abc.abstractmethod
    def forward_train(
        self,
        inputs: InputSample,
        predictions: LabelInstances,
        targets: LabelInstances,
    ) -> LossesType:
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            predictions: Predictions. Batched.
            targets: Targets corresponding to InputSamples.

        Returns:
            LossesType: dict of scalar loss tensors
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self, inputs: InputSample, predictions: LabelInstances
    ) -> PanopticMasks:
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            predictions: Predictions. Batched.

        Returns:
            PanopticMasks: Prediction outputs.
        """
        raise NotImplementedError


def build_panoptic_head(
    cfg: BasePanopticHeadConfig,
) -> BasePanopticHead:  # type: ignore
    """Build a panoptic head from config."""
    registry = RegistryHolder.get_registry(BasePanopticHead)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BasePanopticHead)
        return module
    raise NotImplementedError(f"PanopticHead {cfg.type} not found.")
