"""Backbone interface for Vis4D."""
import abc
from typing import List

import torch
from torch import nn


class BaseModel(nn.Module):
    """Abstract base model for feature extraction."""

    @abc.abstractmethod
    def forward(
        self,
        images: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Base Backbone forward.

        Args:
            images (Tensor[N, C, H, W]): Image input to process. Expected to
                type float32 with values ranging 0..255.

        Returns:
            fp (List[torch.Tensor]): The output feature pyramid. The list index
            represents the level, which has a downsampling raio of 2^index for
            most of the cases. fp[2] is the C2 or P2 in the FPN paper
            (https://arxiv.org/abs/1612.03144). fp[0] is the original image or
            the feature map with the same resolution. fp[1] may be the copy of
            the input image if the network doesn't generate the feature map of
            the resolution.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def out_channels(self) -> List[int]:
        """Get the number of channels for each level of feature pyramid.

        Raises:
            NotImplementedError: _description_

        Returns:
            List[int]: number of channels
        """
        raise NotImplementedError

    def __call__(
        self,
        images: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Type definition for call implementation."""
        return self._call_impl(images)
