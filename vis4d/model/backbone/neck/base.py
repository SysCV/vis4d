"""Base Vis4D neck class."""

import abc

from pydantic import BaseModel, Field

from vis4d.common.module import Vis4DModule
from vis4d.common.registry import RegistryHolder
from vis4d.struct import FeatureMaps


class BaseNeckConfig(BaseModel, extra="allow"):
    """Base config for Neck."""

    type: str = Field(...)


class BaseNeck(Vis4DModule[FeatureMaps, FeatureMaps]):
    """Base Neck class."""

    @abc.abstractmethod
    def __call__(  # type: ignore[override]
        self,
        inputs: FeatureMaps,
    ) -> FeatureMaps:
        """Base Backbone forward.

        Args:
            inputs: Input feature maps (output of backbone).

        Returns:
            FeatureMaps: Dictionary of output feature maps.
        """
        raise NotImplementedError


def build_neck(
    cfg: BaseNeckConfig,
) -> BaseNeck:
    """Build a neck from config."""
    registry = RegistryHolder.get_registry(BaseNeck)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseNeck)
        return module
    raise NotImplementedError(f"Neck {cfg.type} not found.")
