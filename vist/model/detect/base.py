"""Base class for VisT detectors."""

import abc
from typing import Dict, List, Optional, Tuple, Union

import torch

from vist.common.registry import RegistryHolder
from vist.struct import Boxes2D, Images, InputSample, LossesType, ModelOutput

from ..base import BaseModel


class BaseDetector(BaseModel, metaclass=RegistryHolder):
    """Base detector class."""

    @abc.abstractmethod
    def preprocess_image(self, batched_inputs: List[InputSample]) -> Images:
        """Normalize, pad and batch the input images."""
        raise NotImplementedError

    def forward(
        self, batch_inputs: List[List[InputSample]]
    ) -> Union[LossesType, ModelOutput]:
        """Model forward function."""
        if self.training:
            assert all(
                len(inp) == 1 for inp in batch_inputs
            ), "No reference views allowed in detector training!"
            return self.forward_train(batch_inputs)
        inputs = [inp[0] for inp in batch_inputs]  # no ref views during test
        return self.forward_test(inputs)

    @abc.abstractmethod
    def extract_features(self, images: Images) -> Dict[str, torch.Tensor]:
        """Detector feature extraction stage.

        Return backbone output features
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_detections(
        self,
        images: Images,
        features: Dict[str, torch.Tensor],
        proposals: List[Boxes2D],
        targets: Optional[List[Boxes2D]] = None,
        compute_detections: bool = True,
    ) -> Tuple[Optional[List[Boxes2D]], LossesType]:
        """Detector second stage (RoI Head).

        Return losses (empty if no targets) and optionally detections.
        """
        raise NotImplementedError


class BaseTwoStageDetector(BaseDetector):
    """Base class for two-stage detectors."""

    @abc.abstractmethod
    def generate_proposals(
        self,
        images: Images,
        features: Dict[str, torch.Tensor],
        targets: Optional[List[Boxes2D]] = None,
    ) -> Tuple[List[Boxes2D], LossesType]:
        """Detector RPN stage.

        Return proposals per image and losses (empty if no targets).
        """
        raise NotImplementedError
