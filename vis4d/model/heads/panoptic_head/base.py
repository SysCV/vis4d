"""Panoptic Head interface for Vis4D."""

import abc
from typing import List, Optional, Tuple, Union

from torch import nn

from vis4d.struct import InputSample, InstanceMasks, Losses, SemanticMasks

PanopticMasks = Tuple[List[InstanceMasks], List[SemanticMasks]]


class BasePanopticHead(nn.Module):
    """Base Panoptic head class."""

    def forward(  # TODO restructure
        self,
        inputs: InputSample,
        predictions,
        targets=None,
    ) -> Union[Losses, PanopticMasks]:
        """Base Panoptic head forward.

        Args:
            inputs: Model Inputs, batched.
            features: Input feature maps.
            targets: Container with targets, e.g. Boxes2D / 3D, Masks, ...

        Returns:
            Losses / PanopticMasks: In train mode, return losses. In test
            mode, return predictions.
        """
        if targets is not None:  # pragma: no cover
            return self.forward_train(inputs, predictions, targets)
        return self.forward_test(inputs, predictions)

    @abc.abstractmethod
    def forward_train(
        self,
        inputs: InputSample,
        predictions,
        targets,
    ) -> Losses:
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            predictions: Predictions. Batched.
            targets: Targets corresponding to InputSamples.

        Returns:
            Losses: dict of scalar loss tensors
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(self, inputs: InputSample, predictions) -> PanopticMasks:
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            predictions: Predictions. Batched.

        Returns:
            PanopticMasks: Prediction outputs.
        """
        raise NotImplementedError
