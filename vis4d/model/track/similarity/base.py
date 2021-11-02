"""Vis4D base class for similarity networks."""

import abc
from typing import Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field

from vis4d.common.bbox.samplers import SamplingResult
from vis4d.common.registry import RegistryHolder
from vis4d.struct import Boxes2D, InputSample, LossesType


class SimilarityLearningConfig(BaseModel, extra="allow"):
    """Base config for similarity learning."""

    type: str = Field(...)


class BaseSimilarityHead(torch.nn.Module, metaclass=RegistryHolder):  # type: ignore # pylint: disable=line-too-long
    """Base similarity learning head class."""

    @abc.abstractmethod
    def forward_train(
        self,
        inputs: List[InputSample],
        features: List[Dict[str, torch.Tensor]],
        boxes: List[List[Boxes2D]],
    ) -> Tuple[LossesType, Optional[List[SamplingResult]]]:
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched, including
                possible reference views. The keyframe is at index 0.
            features: Input feature maps. Batched, including possible
                reference views. The keyframe is at index 0.
            boxes: Detected boxes to apply similarity learning on.

        Returns:
            LossesType: A dict of scalar loss tensors.
            Optional[List[SamplingResult]]: Sampling results. Key first, then
                reference views.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self,
        inputs: InputSample,
        features: Dict[str, torch.Tensor],
        boxes: List[Boxes2D],
    ) -> List[torch.Tensor]:
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            boxes: Input boxes to compute similarity embedding for.

        Returns:
            List[torch.Tensor]: Similarity embeddings (one vector per box, one
            tensor per batch element).
        """
        raise NotImplementedError


def build_similarity_head(cfg: SimilarityLearningConfig) -> BaseSimilarityHead:
    """Build a SimilarityHead from config."""
    registry = RegistryHolder.get_registry(BaseSimilarityHead)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseSimilarityHead)
        return module
    raise NotImplementedError(f"RoIHead {cfg.type} not found.")
