"""RoI Pooling module base."""
import abc
from typing import List, Tuple

import torch
from torch import nn


class RoIPooler(nn.Module):
    """Base class for RoI poolers."""

    def __init__(self, resolution: Tuple[int, int]) -> None:
        """Init."""
        super().__init__()
        self.resolution = resolution

    @abc.abstractmethod
    def forward(
        self, features: List[torch.Tensor], boxes: List[torch.Tensor]
    ) -> torch.Tensor:
        """Pool features in input bounding boxes from given feature maps."""
        raise NotImplementedError
