"""RoI Head interface for Vis4D."""
import abc
from typing import Dict, List, Optional, Tuple, Union, overload

from pydantic import BaseModel, Field

from vis4d.common.module import TTestReturn, TTrainReturn, Vis4DModule
from vis4d.common.registry import RegistryHolder
from vis4d.struct import (
    Boxes2D,
    FeatureMaps,
    InputSample,
    LabelInstances,
    LossesType,
)


class BaseRoIHeadConfig(BaseModel, extra="allow"):
    """Base config for RoI head."""

    type: str = Field(...)
    category_mapping: Optional[Dict[str, int]] = None


class BaseRoIHead(Vis4DModule[Tuple[LossesType, TTrainReturn], TTestReturn]):
    """Base RoI head class."""

    @overload  # type: ignore[override]
    def __call__(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        features: FeatureMaps,
    ) -> TTestReturn:  # noqa: D102
        ...

    @overload
    def __call__(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, TTrainReturn]:
        ...

    def __call__(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        features: FeatureMaps,
        targets: Optional[LabelInstances] = None,
    ) -> Union[Tuple[LossesType, TTrainReturn], TTestReturn]:
        """Base RoI head forward.

        Args:
            inputs: Model Inputs, batched.
            boxes: 2D boxes that serve as basis for RoI sampling / pooling.
            features: Input feature maps.
            targets: Container with targets, e.g. Boxes2D / 3D, Masks, ...

        Returns:
            Tuple[LossesType, TTrainReturn]
            or TTestReturn: In train mode, return losses and optionally
            intermediate returns. In test mode, return predictions.
        """
        if targets is not None:
            return self.forward_train(inputs, boxes, features, targets)
        return self.forward_test(inputs, boxes, features)

    @abc.abstractmethod
    def forward_train(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, TTrainReturn]:
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            boxes: Input boxes to apply RoIHead on.
            features: Input feature maps. Batched.
            targets: Targets corresponding to InputSamples.

        Returns:
            LossesType: A dict of scalar loss tensors.
            TTrainReturn: Some intermediate results.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        features: FeatureMaps,
    ) -> TTestReturn:
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            boxes: Input boxes to apply RoIHead on.
            features: Input feature maps. Batched.

        Returns:
            TTestReturn: Prediction output.
        """
        raise NotImplementedError


def build_roi_head(cfg: BaseRoIHeadConfig) -> BaseRoIHead:  # type: ignore
    """Build a roi head from config."""
    registry = RegistryHolder.get_registry(BaseRoIHead)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseRoIHead)
        return module
    raise NotImplementedError(f"RoIHead {cfg.type} not found.")
