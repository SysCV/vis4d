"""RoI Head interface for Vis4D."""
import abc
from typing import Dict, List, Optional, Tuple, Union, overload

from vis4d.common.bbox.samplers import SamplingResult
from vis4d.common.module import TTestReturn, TTrainReturn, Vis4DModule
from vis4d.struct import (
    Boxes2D,
    Boxes3D,
    FeatureMaps,
    InputSample,
    InstanceMasks,
    Losses,
)


class BaseRoIHead(Vis4DModule[Tuple[Losses, TTrainReturn], TTestReturn]):
    """Base RoI head class."""

    def __init__(
        self, category_mapping: Optional[Dict[str, int]] = None
    ) -> None:
        """Init."""
        super().__init__()
        self.category_mapping = category_mapping

    @overload  # type: ignore[override]
    def __call__(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        boxes: List[Boxes2D],
    ) -> TTestReturn:  # noqa: D102
        ...

    @overload
    def __call__(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        boxes: List[Boxes2D],
        targets,
    ) -> Tuple[Losses, TTrainReturn]:
        ...

    def __call__(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        boxes: List[Boxes2D],
        targets=None,
    ) -> Union[Tuple[Losses, TTrainReturn], TTestReturn]:
        """Base RoI head forward.

        Args:
            inputs: Model Inputs, batched.
            features: Input feature maps.
            boxes: 2D boxes that serve as basis for RoI sampling / pooling.
            targets: Container with targets, e.g. Boxes2D / 3D, Masks, ...

        Returns:
            Tuple[LossesType, TTrainReturn]
            or TTestReturn: In train mode, return losses and optionally
            intermediate returns. In test mode, return predictions.
        """
        if targets is not None:
            return self.forward_train(inputs, features, boxes, targets)
        return self.forward_test(inputs, features, boxes)

    @abc.abstractmethod
    def forward_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        boxes: List[Boxes2D],
        targets,
    ) -> Tuple[Losses, TTrainReturn]:
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            boxes: Input boxes to apply RoIHead on.
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
        features: FeatureMaps,
        boxes: List[Boxes2D],
    ) -> TTestReturn:
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            boxes: Input boxes to apply RoIHead on.

        Returns:
            TTestReturn: Prediction output.
        """
        raise NotImplementedError


Det2DRoIHead = BaseRoIHead[
    Optional[SamplingResult],
    Tuple[List[Boxes2D], Optional[List[InstanceMasks]]],
]
Det3DRoIHead = BaseRoIHead[SamplingResult, List[Boxes3D]]
