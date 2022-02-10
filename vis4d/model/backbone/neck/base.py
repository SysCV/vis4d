"""Base Vis4D neck class."""

from vis4d.common import Vis4DModule
from vis4d.struct import FeatureMaps


class BaseNeck(Vis4DModule[FeatureMaps, FeatureMaps]):
    """Base Neck class."""

    def __call__(  # type: ignore[override]
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
