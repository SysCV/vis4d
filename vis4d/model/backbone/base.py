"""Backbone interface for Vis4D."""

import abc
from typing import Optional

from pydantic import BaseModel, Field

from vis4d.common.module import Vis4DModule
from vis4d.common.registry import RegistryHolder
from vis4d.struct import FeatureMaps, InputSample

from .neck import BaseNeckConfig


class BaseBackboneConfig(BaseModel, extra="allow"):
    """Base config for Backbone."""

    type: str = Field(...)
    neck: Optional[BaseNeckConfig]


class BaseBackbone(Vis4DModule[FeatureMaps, FeatureMaps]):
    """Base Backbone class."""

    @abc.abstractmethod
    def preprocess_inputs(self, inputs: InputSample) -> InputSample:
        """Normalize the input images."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(  # type: ignore[override]
        self,
        inputs: InputSample,
    ) -> FeatureMaps:
        """Base Backbone forward.

        Args:
            inputs: Model Inputs, batched.

        Returns:
            FeatureMaps: Dictionary of output feature maps.
        """
        raise NotImplementedError


def build_backbone(
    cfg: BaseBackboneConfig,
) -> BaseBackbone:
    """Build a backbone from config."""
    registry = RegistryHolder.get_registry(BaseBackbone)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseBackbone)
        return module
    raise NotImplementedError(f"Backbone {cfg.type} not found.")
