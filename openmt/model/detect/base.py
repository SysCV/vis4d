"""Base class for openMT detectors."""

import abc
from typing import Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel

from openmt.common.registry import RegistryHolder
from openmt.struct import Boxes2D, DetectionOutput, ImageList


class BaseDetectorConfig(BaseModel, extra="allow"):
    """Config for detection detect."""

    type: str


class BaseDetector(torch.nn.Module, metaclass=RegistryHolder):  # type: ignore
    """Base detector class."""

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        """Get device where detect input should be moved to."""
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess_image(
        self, batched_inputs: Tuple[Dict[str, torch.Tensor]]
    ) -> ImageList:
        """Normalize, pad and batch the input images."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        inputs: ImageList,
        targets: Optional[List[Boxes2D]] = None,
    ) -> DetectionOutput:
        """Detector forward function.

        Return backbone output features, proposals, detections and optionally
        training losses.
        """
        raise NotImplementedError


def build_detector(cfg: BaseDetectorConfig) -> BaseDetector:
    """Build a detector.

    Note that it does not load any weights from ``cfg``.
    """
    assert cfg is not None
    registry = RegistryHolder.get_registry(__package__)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseDetector)
        return module
    raise NotImplementedError(f"Detector {cfg.type} not found.")
