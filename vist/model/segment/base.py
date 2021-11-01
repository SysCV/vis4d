"""Base class for VisT segmentors."""

import abc
from typing import Dict, List, Optional, Tuple

import torch

from vist.common.registry import RegistryHolder
from vist.struct import InputSample, LossesType, SemanticMasks

from ..base import BaseModel


class BaseSegmentor(BaseModel, metaclass=RegistryHolder):
    """Base segmentor class."""

    @abc.abstractmethod
    def preprocess_inputs(self, inputs: List[InputSample]) -> InputSample:
        """Normalize, pad and batch input images. Preprocess other inputs."""
        raise NotImplementedError

    @abc.abstractmethod
    def extract_features(self, inputs: InputSample) -> Dict[str, torch.Tensor]:
        """Segmentor feature extraction stage.

        Return backbone output features
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_segmentations(
        self,
        inputs: InputSample,
        features: Dict[str, torch.Tensor],
        compute_segmentations: bool = True,
    ) -> Tuple[Optional[List[SemanticMasks]], LossesType]:
        """Segmentor decode stage.

        Return losses (empty if not training) and optionally segmentations.
        """
        raise NotImplementedError
