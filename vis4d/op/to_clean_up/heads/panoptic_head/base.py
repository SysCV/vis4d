"""Panoptic Head interface for Vis4D."""

import abc
from typing import List, Optional, Tuple, Union

from torch import nn

from vis4d.struct_to_revise import (
    InputSample,
    InstanceMasks,
    LabelInstances,
    LossesType,
    SemanticMasks,
)

PanopticMasks = Tuple[List[InstanceMasks], List[SemanticMasks]]


class BasePanopticHead(nn.Module):
    """Base Panoptic head class."""

    def forward(  # TODO restructure
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
        if targets is not None:  # pragma: no cover
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