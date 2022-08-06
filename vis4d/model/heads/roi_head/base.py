"""RoI Head interface for Vis4D."""
import abc
from typing import Dict, List, Optional, Tuple, Union

from torch import nn

from vis4d.struct import Boxes2D, FeatureMaps, InputSample, Losses


class BaseRoIHead(nn.Module):
    """Base RoI head class."""

    def __init__(
        self, category_mapping: Optional[Dict[str, int]] = None
    ) -> None:
        """Init."""
        super().__init__()
        self.category_mapping = category_mapping

    def forward(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        boxes: List[Boxes2D],
        targets=None,
    ):
        """Base RoI head forward.

        Args:
            inputs: Model Inputs, batched.
            features: Input feature maps.
            boxes: 2D boxes that serve as basis for RoI sampling / pooling.
            targets: Container with targets, e.g. Boxes2D / 3D, Masks, ...

        Returns:
            Tuple[Losses, TTrainReturn]
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
    ):
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            boxes: Input boxes to apply RoIHead on.
            targets: Targets corresponding to InputSamples.

        Returns:
            Losses: A dict of scalar loss tensors.
            TTrainReturn: Some intermediate results.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        boxes: List[Boxes2D],
    ):
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            boxes: Input boxes to apply RoIHead on.

        Returns:
            TTestReturn: Prediction output.
        """
        raise NotImplementedError
