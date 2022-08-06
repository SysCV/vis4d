"""Base Vis4D neck class."""
from torch import nn

from vis4d.struct import NamedTensors


class BaseNeck(nn.Module):
    """Base Neck class."""

    def forward(
        self,
        inputs: NamedTensors,
    ) -> NamedTensors:
        """Base Neck forward.

        Args:
            inputs: Input feature maps (output of backbone).

        Returns:
            NamedTensors: Dictionary of output feature maps.
        """
        raise NotImplementedError
