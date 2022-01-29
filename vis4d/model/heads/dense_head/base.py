"""Dense Head interface for Vis4D."""
import abc
from typing import Dict, List, Optional, Tuple, Union, overload

import torch

from vis4d.common.module import TTestReturn, TTrainReturn, Vis4DModule
from vis4d.struct import (
    Boxes2D,
    FeatureMaps,
    InputSample,
    LabelInstances,
    LossesType,
    MaskLogits,
    SemanticMasks,
)


class BaseDenseHead(Vis4DModule[Tuple[LossesType, TTrainReturn], TTestReturn]):
    """Base Dense head class."""

    def __init__(
        self, category_mapping: Optional[Dict[str, int]] = None
    ) -> None:
        """Init."""
        super().__init__()
        self.category_mapping = category_mapping

    @overload  # type: ignore[override]
    def __call__(
        self, inputs: InputSample, features: FeatureMaps
    ) -> TTestReturn:  # noqa: D102
        ...

    @overload
    def __call__(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, TTrainReturn]:
        ...

    def __call__(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: Optional[LabelInstances] = None,
    ) -> Union[Tuple[LossesType, TTrainReturn], TTestReturn]:
        """Base Dense head forward.

        Args:
            inputs: Model Inputs, batched.
            features: Input feature maps.
            targets: Container with targets, e.g. Boxes2D / 3D, Masks, ...

        Returns:
            [LossesType, TTrainReturn] / TTestReturn: In train mode, return
            losses and intermediate outputs. In test mode, return predictions.
        """
        if targets is not None:
            return self.forward_train(inputs, features, targets)
        return self.forward_test(inputs, features)

    @abc.abstractmethod
    def forward_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, TTrainReturn]:
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            targets: Targets corresponding to InputSamples.

        Returns:
            Tuple[LossesType, TTrainReturn]: Tuple of:
             (dict of scalar loss tensors, predictions / other outputs)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self, inputs: InputSample, features: FeatureMaps
    ) -> TTestReturn:
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.

        Returns:
            TTestReturn: Prediction output.
        """
        raise NotImplementedError


ClsDenseHead = BaseDenseHead[Optional[List[torch.Tensor]], List[torch.Tensor]]
DetDenseHead = BaseDenseHead[List[Boxes2D], List[Boxes2D]]
SegDenseHead = BaseDenseHead[Optional[List[MaskLogits]], List[SemanticMasks]]
