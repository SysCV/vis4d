"""RoIHead interface for backend."""

import abc
from typing import Dict, List, Optional, Tuple

import torch
from detectron2.structures import ImageList  # TODO override with our own?

from openmt.config import RoIHead
from openmt.core.registry import RegistryHolder
from openmt.structures import Boxes2D


class BaseRoIHead(torch.nn.Module, metaclass=RegistryHolder):
    @abc.abstractmethod
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Boxes2D],
        targets: Optional[List[Boxes2D]] = None,
    ) -> Tuple[List[Boxes2D], Optional[List[Boxes2D]]]:  # TODO signature
        """Process proposals and output predictions and possibly target
        assignments."""
        raise NotImplementedError


def build_roi_head(cfg: RoIHead) -> BaseRoIHead:
    """Build an RoIHead from config."""
    model_registry = RegistryHolder.get_registry(__package__)
    if cfg.type in model_registry:
        return model_registry[cfg.type](cfg)
    else:
        raise NotImplementedError(f"RoIHead {cfg.type} not found.")
