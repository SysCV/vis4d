"""RoI Head interface for Vis4D."""
import abc
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from vis4d.struct import (
    Boxes2D,
    InputSample,
    LabelInstances,
    LossesType,
    NamedTensors,
    TTestReturn,
    TTrainReturn,
)


class BaseRoIBox2DHead(nn.Module):
    """Base RoI head class."""

    def forward(
        self,
        features: List[torch.Tensor],
        boxes: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Base RoI head forward.

        Args:
            features: Input feature maps (N, C, H/s, W/s) with different s.
            boxes: 2D boxes that serve as basis for RoI sampling / pooling.
            targets: Container with targets, e.g. Boxes2D / 3D, Masks, ...

        Returns:
            boxes (M, 4)
            scores (M,)
            class_ids (M,)
        """
        raise NotImplementedError
