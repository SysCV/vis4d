"""Dense Head interface for Vis4D."""
import abc
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from vis4d.struct import Boxes2D, NamedTensors, LossesType


class BaseDenseBox2DHead(nn.Module, abc.ABC):
    """Base Box2D head class."""

    def __init__(
        self, category_mapping: Optional[Dict[str, int]] = None
    ) -> None:
        """Init."""
        super().__init__()
        self.category_mapping = category_mapping

    @abc.abstractmethod
    def forward(
        self,
        features: NamedTensors,
    ) -> Tuple[NamedTensors, NamedTensors]:
        """Base Box2D head forward.

        Args:
            features (Dict[Tensor]): Input feature maps.

        Returns:
            Tuple[NamedTensors, NamedTensors]: Class scores and box
                regression parameters per image.
        """
        pass

    @abc.abstractmethod
    def postprocess(
        self, class_outs: NamedTensors, regression_outs: NamedTensors
    ) -> List[Boxes2D]:
        """Box2D head postprocessing.

        Args:
            class_outs (Dict[Tensor]): Class scores TODO finish
            class_outs (Dict[Tensor]): Regression parameters per

        Returns:
            List[Boxes2D]: Output boxes after postprocessing.
        """
        pass

    @abc.abstractmethod
    def loss(
        self,
        class_outs: NamedTensors,
        regression_outs: NamedTensors,
        targets: List[Boxes2D],
        images_shape: Tuple[int, int, int, int],
    ) -> LossesType:
        """Loss computation.

        Args:
            outputs: Network outputs.
            targets (List[Boxes2D]): Target 2D boxes.
            metadata (Dict): Dictionary of metadata needed for loss, e.g.
                image size, feature map strides, etc.
        Returns:
            LossesType: Dictionary of scalar loss tensors.
        """
        pass


class BaseSegmentationHead(nn.Module):
    # TODO
    ...
