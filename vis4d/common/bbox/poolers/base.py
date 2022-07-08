"""RoI Pooling module base."""
import abc
from typing import List, Tuple

import torch
from torch import nn

from vis4d.struct import Boxes2D

# TODO should be abstract?


class BaseRoIPooler(nn.Module):
    """Base class for RoI poolers."""

    def __init__(self, resolution: Tuple[int, int]) -> None:
        """Init."""
        super().__init__()
        self.resolution = resolution

    @abc.abstractmethod
    def forward(
        self, features: List[torch.Tensor], boxes: List[Boxes2D]
    ) -> torch.Tensor:
        """Pool features in input bounding boxes from given feature maps."""
        raise NotImplementedError
