"""Base class for VisT detectors."""

import abc
from typing import Dict, List, Optional, Tuple

import torch

from vist.common.registry import RegistryHolder
from vist.struct import Boxes2D, InputSample, InstanceMasks, LossesType

from ..base import BaseModel


class BaseDetector(BaseModel, metaclass=RegistryHolder):
    """Base detector class."""

    @abc.abstractmethod
    def preprocess_inputs(self, inputs: List[InputSample]) -> InputSample:
        """Normalize, pad and batch input images. Preprocess other inputs."""
        raise NotImplementedError

    @abc.abstractmethod
    def extract_features(self, inputs: InputSample) -> Dict[str, torch.Tensor]:
        """Detector feature extraction stage.

        Return backbone output features
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_detections(
        self,
        inputs: InputSample,
        features: Dict[str, torch.Tensor],
        proposals: Optional[List[Boxes2D]] = None,
        compute_detections: bool = True,
        compute_segmentations: bool = False,
    ) -> Tuple[
        Optional[List[Boxes2D]], LossesType, Optional[List[InstanceMasks]]
    ]:
        """Detector second stage (RoI Head).

        Return losses (empty if not training) and optionally detections.
        """
        raise NotImplementedError


class BaseTwoStageDetector(BaseDetector):
    """Base class for two-stage detectors."""

    @abc.abstractmethod
    def generate_proposals(
        self,
        inputs: InputSample,
        features: Dict[str, torch.Tensor],
    ) -> Tuple[List[Boxes2D], LossesType]:
        """Detector RPN stage.

        Return proposals per image and losses (empty if no targets).
        """
        raise NotImplementedError
