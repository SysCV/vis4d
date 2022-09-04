"""Feature pyramid processing base class."""
import abc

from typing import List

import torch
from torch import nn


class FeaturePyramidProcessing(nn.Module):
    """Base Neck class."""

    @abc.abstractmethod
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Feature pyramid processing.

        This module do a further processing for the hierarchical feature
        representation extracted by the base models.

        Args:
            x (List[torch.Tensor]): Feature pyramid as outputs of the base
            model.

        Returns:
            List[torch.Tensor]: Feature pyramid after the processing.
        """
        raise NotImplementedError

    def __call__(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Type definition for call implementation."""
        return self._call_impl(x)
