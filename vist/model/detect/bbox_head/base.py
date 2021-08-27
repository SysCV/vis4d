"""3D Head interface for backend."""

import abc
from typing import Any

import torch
from pydantic import BaseModel, Field

from vist.common.registry import RegistryHolder


class BaseBoundingBoxConfig(BaseModel, extra="allow"):
    """Base config for bbox head."""

    type: str = Field(...)


class BaseBoundingBoxHead(torch.nn.Module, metaclass=RegistryHolder):  # type: ignore # pylint: disable=line-too-long
    """Base Bounding Box head class."""

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore
        """Forward method.

        Process proposals, output the predictions of 2D / 3D bboxes.
        """
        raise NotImplementedError


def build_bbox_head(cfg: BaseBoundingBoxConfig) -> BaseBoundingBoxHead:
    """Build a bbox head from config."""
    registry = RegistryHolder.get_registry(BaseBoundingBoxHead)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseBoundingBoxHead)
        return module
    raise NotImplementedError(f"BBox Head {cfg.type} not found.")
