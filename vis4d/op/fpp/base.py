"""Feature pyramid processing base class."""
from __future__ import annotations

import abc

import torch
from torch import nn


class FeaturePyramidProcessing(nn.Module):
    """Base Neck class."""

    @abc.abstractmethod
    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Feature pyramid processing.

        This module do a further processing for the hierarchical feature
        representation extracted by the base models.

        Args:
            features (list[torch.Tensor]): Feature pyramid as outputs of the
            base model.

        Returns:
            list[torch.Tensor]: Feature pyramid after the processing.
        """
        raise NotImplementedError

    def __call__(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Type definition for call implementation."""
        return self._call_impl(features)
