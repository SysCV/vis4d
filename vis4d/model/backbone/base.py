"""Backbone interface for Vis4D."""

import abc
from typing import List, Optional, Tuple

import torch
from pydantic import BaseModel, Field

from vis4d.common.module import Vis4DModule
from vis4d.common.registry import RegistryHolder
from vis4d.struct import FeatureMaps, InputSample

from .neck import BaseNeckConfig


class BaseBackboneConfig(BaseModel, extra="allow"):
    """Base config for Backbone."""

    type: str = Field(...)
    pixel_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    pixel_std: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    output_names: Optional[List[str]]
    neck: Optional[BaseNeckConfig]


class BaseBackbone(Vis4DModule[FeatureMaps, FeatureMaps]):
    """Base Backbone class."""

    def __init__(self, cfg: BaseBackboneConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg = cfg

        self.register_buffer(
            "pixel_mean",
            torch.tensor(self.cfg.pixel_mean).view(-1, 1, 1),
            False,
        )
        self.register_buffer(
            "pixel_std", torch.tensor(self.cfg.pixel_std).view(-1, 1, 1), False
        )

    def preprocess_inputs(self, inputs: InputSample) -> InputSample:
        """Normalize the input images."""
        inputs.images.tensor = (
            inputs.images.tensor - self.pixel_mean
        ) / self.pixel_std
        return inputs

    def get_outputs(self, outs: List[torch.Tensor]) -> FeatureMaps:
        """Get feature map dict."""
        if self.cfg.output_names is None:
            backbone_outs = {f"out{i}": v for i, v in enumerate(outs)}
        else:  # pragma: no cover
            assert len(self.cfg.output_names) == len(outs)
            backbone_outs = dict(zip(self.cfg.output_names, outs))
        return backbone_outs

    @abc.abstractmethod
    def __call__(  # type: ignore[override]
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
