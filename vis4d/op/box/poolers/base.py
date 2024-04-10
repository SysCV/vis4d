"""RoI Pooling module base."""

from __future__ import annotations

import abc

import torch
from torch import nn


class RoIPooler(nn.Module):
    """Base class for RoI poolers."""

    def __init__(self, resolution: tuple[int, int]) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.resolution = resolution

    @abc.abstractmethod
    def forward(
        self, features: list[torch.Tensor], boxes: list[torch.Tensor]
    ) -> torch.Tensor:
        """Pool features in input bounding boxes from given feature maps."""
        raise NotImplementedError
