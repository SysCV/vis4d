"""VisT base class for similarity networks."""

import abc
from typing import Dict, List, Optional, Tuple, Union

import torch
from pydantic import BaseModel, Field

from vist.common.bbox.samplers import SamplingResult
from vist.common.registry import RegistryHolder
from vist.struct import Boxes2D, Images, LossesType


class SimilarityLearningConfig(BaseModel, extra="allow"):
    """Base config for similarity learning."""

    type: str = Field(...)


class BaseSimilarityHead(torch.nn.Module, metaclass=RegistryHolder):  # type: ignore # pylint: disable=line-too-long
    """Base similarity learning head class."""

    @abc.abstractmethod
    def forward_train(
        self,
        inputs: Union[List[Images], List[Dict[str, torch.Tensor]]],
        boxes: List[List[Boxes2D]],
        targets: List[List[Boxes2D]],
    ) -> Tuple[LossesType, Optional[List[SamplingResult]]]:
        """Forward pass during training stage.

        Args:
            inputs: Either images or feature maps. Batched, including possible
                reference views.
            boxes: Detected boxes to apply similarity learning on.
            targets: Target boxes with tracking identities.

        Returns:
            LossesType: A dict of scalar loss tensors.
             Optional[List[SamplingResult]]: Sampling result for
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self,
        inputs: Union[Images, Dict[str, torch.Tensor]],
        boxes: List[Boxes2D],
    ) -> List[torch.Tensor]:
        """Forward pass during testing stage.

        Args:
            inputs: Model input (batched).
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
