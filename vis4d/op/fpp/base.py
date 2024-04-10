"""Feature pyramid processing base class."""

from __future__ import annotations

import abc

from torch import Tensor, nn


class FeaturePyramidProcessing(nn.Module):
    """Base Neck class."""

    @abc.abstractmethod
    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """Feature pyramid processing.

        This module do a further processing for the hierarchical feature
        representation extracted by the base models.

        Args:
            features (list[Tensor]): Feature pyramid as outputs of the
            base model.

        Returns:
            list[Tensor]: Feature pyramid after the processing.
        """
        raise NotImplementedError

    def __call__(self, features: list[Tensor]) -> list[Tensor]:
        """Type definition for call implementation."""
        return self._call_impl(features)
