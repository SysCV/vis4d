"""Base class for Vis4D segmentation models."""

import abc
from typing import List

import torch
from torch import nn


class BaseSegmentor(nn.Module):
    """Base segmentation head class."""

    @abc.abstractmethod
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass during training stage.

        Args:
            x (List[torch.Tensor]): Multi-level features.

        Returns:
            predictions (List[torch.Tensor]): Pixel-level segmentation
                predictions.
        """
        raise NotImplementedError

    def __call__(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        return super()._call_impl(x)
