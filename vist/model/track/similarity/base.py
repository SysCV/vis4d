"""RoIHead interface for backend."""

import abc
from typing import Any

import torch
from pydantic import BaseModel, Field

from vist.common.registry import RegistryHolder


class SimilarityLearningConfig(BaseModel, extra="allow"):
    """Base config for similarity learning."""

    type: str = Field(...)


class BaseSimilarityHead(torch.nn.Module, metaclass=RegistryHolder):  # type: ignore # pylint: disable=line-too-long
    """Base similarity head class."""

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore
        """Forward method.

        Process proposals, output predictions and possibly target
        assignments.
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
