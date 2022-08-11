"""Backbone interface for Vis4D."""
import abc
from typing import List, Optional, Tuple

import torch
from torch import nn

from .neck import BaseNeck


class BaseBackbone(nn.Module):
    """Base Backbone class."""

    def __init__(
        self,
        pixel_mean: Tuple[float, float, float],
        pixel_std: Tuple[float, float, float],
        out_indices: Optional[List[int]] = None,
        neck: Optional[BaseNeck] = None,
    ) -> None:
        """Init BaseBackbone."""
        super().__init__()
        self.out_indices = out_indices
        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False
        )
        self.neck = neck

    def preprocess_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize the input images."""
        return (inputs - self.pixel_mean) / self.pixel_std

    def get_outputs(self, outs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Get feature map dict."""
        if self.out_indices is not None:
            outs = [outs[ind] for ind in self.out_indices]
        return outs

    @abc.abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Base Backbone forward.

        Args:
            inputs (Tensor[N, C, H, W]): Image input to process. Expected to
                type float32 with vlaues ranging 0..255.

        Returns:
            NamedTensors (Dict[Tensor]): output feature maps.
        """
        raise NotImplementedError
