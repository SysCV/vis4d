"""Backbone interface for Vis4D."""
import abc
from typing import List, Tuple

import torch
from torch import nn


class BaseBackbone(nn.Module):
    """Base Backbone class."""

    def __init__(
        self,
        pixel_mean: Tuple[float, float, float],
        pixel_std: Tuple[float, float, float],
    ) -> None:
        """Init BaseBackbone."""
        super().__init__()
        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False
        )

    def preprocess_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize the input images."""
        return (inputs - self.pixel_mean) / self.pixel_std

    @abc.abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Base Backbone forward.

        Args:
            inputs (Tensor[N, C, H, W]): Image input to process. Expected to
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
