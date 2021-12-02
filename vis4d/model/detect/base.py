"""Base class for Vis4D detectors."""

import abc
from typing import List, Optional, Tuple

from vis4d.common.registry import RegistryHolder
from vis4d.struct import (
    Boxes2D,
    FeatureMaps,
    InputSample,
    InstanceMasks,
    LabelInstances,
    LossesType,
)

from ..base import BaseModel, BaseModelConfig


class BaseDetectorConfig(BaseModelConfig):
    """Base configuration for detectors."""

    clip_bboxes_to_image: bool = True


class BaseDetector(BaseModel, metaclass=RegistryHolder):
    """Base detector class."""

    def __init__(self, cfg: BaseModelConfig):
        """Init."""
        super().__init__(cfg)
        self.cfg: BaseDetectorConfig = BaseDetectorConfig(**cfg.dict())

    @abc.abstractmethod
    def extract_features(self, inputs: InputSample) -> FeatureMaps:
        """Detector feature extraction stage.

        Return backbone output features
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_detections(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        proposals: Optional[List[Boxes2D]] = None,
        targets: Optional[LabelInstances] = None,
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
        features: FeatureMaps,
        targets: Optional[LabelInstances] = None,
    ) -> Tuple[List[Boxes2D], LossesType]:
        """Detector RPN stage.

        Return proposals per image and losses (empty if no targets).
        """
        raise NotImplementedError
