"""Semantic segmentation dead interface for Vis4D."""

import abc
from typing import List

import torch
from torch import nn


class BaseSegmentHead(nn.Module):
    """Base segmentation head class."""

    @abc.abstractmethod
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass during training stage.

        Args:
            x (List[torch.Tensor]): Multi-level features.

        Returns:
            predictions (torch.Tensor): Pixel-level segmentation predictions.
        """
        raise NotImplementedError
