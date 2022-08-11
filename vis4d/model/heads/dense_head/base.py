"""Dense Head interface for Vis4D."""
import abc
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from vis4d.struct import Boxes2D, LossesType, NamedTensors

# TODO Tobias: remove


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
        features: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor],]:
        """Base Box2D head forward.

        Args:
            features (List[Tensor]): Input feature maps (N, C, H/s, W/s) at
                different strides.

        Returns:
            boxes (after postprocessing), scores, class_ids,
            classication_outputs, regression_outputs  # TODO revisit

        """
        raise NotImplementedError


# TODO make abstract


class BaseDenseBox2DHeadLoss(nn.Module):
    """mmdet dense head loss wrapper."""

    def loss(
        self,
        class_outs: NamedTensors,
        regression_outs: NamedTensors,
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
        images_shape: Tuple[int, int, int, int],
    ) -> LossesType:
        raise NotImplementedError


class TransformDenseHeadOutputs(nn.Module):
    """Convert classification output, regression output into Boxes of form x1y1x2y2 score, [class id]."""

    def forward(
        self,
        class_outs: List[torch.Tensor],
        regression_outs: List[torch.Tensor],
        images_shape: Tuple[int, int, int, int],
    ) -> List[Boxes2D]:
        raise NotImplementedError


class BaseSegmentationHead(nn.Module):
    # TODO
    ...
