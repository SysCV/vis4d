"""Base model interface."""

from __future__ import annotations

import abc

import torch
from torch import nn


class BaseModel(nn.Module):
    """Abstract base model for feature extraction."""

    @abc.abstractmethod
    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Base model forward.

        Args:
            images (Tensor[N, C, H, W]): Image input to process. Expected to be
                type float32.

        Raises:
            NotImplementedError: This is an abstract class method.

        Returns:
            fp (list[torch.Tensor]): The output feature pyramid. The list index
            represents the level, which has a downsampling ratio of 2^index for
            most of the cases. fp[2] is the C2 or P2 in the FPN paper
            (https://arxiv.org/abs/1612.03144). fp[0] is the original image or
            the feature map with the same resolution. fp[1] may be the copy of
            the input image if the network doesn't generate the feature map of
            the resolution.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def out_channels(self) -> list[int]:
        """Get the number of channels for each level of feature pyramid.

        Raises:
            NotImplementedError: This is an abstract class method.

        Returns:
            list[int]: Number of channels.
        """
        raise NotImplementedError

    def __call__(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Type definition for call implementation.

        Args:
            images (torch.Tensor): Image input to process.

        Returns:
            list[torch.Tensor]: The output feature pyramid.
        """
        return self._call_impl(images)
