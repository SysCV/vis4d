"""RoI Head interface for VisT."""

import abc
<<<<<<< HEAD
from typing import Dict, List, Optional, Tuple
=======
from typing import Dict, List, Optional, Sequence, Tuple
>>>>>>> main

import torch
from pydantic import BaseModel, Field

from vist.common.bbox.samplers import SamplingResult
from vist.common.registry import RegistryHolder
from vist.struct import Boxes2D, InputSample, LabelInstance, LossesType


class BaseRoIHeadConfig(BaseModel, extra="allow"):
    """Base config for RoI head."""

    type: str = Field(...)


class BaseRoIHead(torch.nn.Module, metaclass=RegistryHolder):  # type: ignore
    """Base RoI head class."""

    @abc.abstractmethod
    def forward_train(
        self,
        inputs: InputSample,
        features: Dict[str, torch.Tensor],
        boxes: List[Boxes2D],
    ) -> Tuple[LossesType, Optional[SamplingResult]]:
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            boxes: Input boxes to apply RoIHead on.

        Returns:
            LossesType: A dict of scalar loss tensors.
            Optional[List[SamplingResult]]: Sampling result.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self,
        inputs: InputSample,
        features: Dict[str, torch.Tensor],
        boxes: List[Boxes2D],
<<<<<<< HEAD
    ) -> List[LabelInstance]:
=======
    ) -> Sequence[LabelInstance]:
>>>>>>> main
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            boxes: Input boxes to apply RoIHead on.

        Returns:
            List[LabelInstance]: Prediction output.
        """
        raise NotImplementedError


def build_roi_head(cfg: BaseRoIHeadConfig) -> BaseRoIHead:
    """Build a roi head from config."""
    registry = RegistryHolder.get_registry(BaseRoIHead)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseRoIHead)
        return module
    raise NotImplementedError(f"RoIHead {cfg.type} not found.")
