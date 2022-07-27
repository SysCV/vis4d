"""Base Vis4D neck class."""
from torch import nn

from vis4d.struct import FeatureMaps


class BaseNeck(nn.Module):
    """Base Neck class."""

    def forward(
        self,
        inputs: FeatureMaps,
    ) -> FeatureMaps:
        """Base Neck forward.

        Args:
            inputs: Input feature maps (output of backbone).

        Returns:
            FeatureMaps: Dictionary of output feature maps.
        """
        raise NotImplementedError
