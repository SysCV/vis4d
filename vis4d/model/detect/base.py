"""Base class for Vis4D detectors."""

import abc
from typing import List, Optional, Tuple, Union, overload

from vis4d.common.bbox.samplers import SamplingResult
from vis4d.common.module import TTestReturn, TTrainReturn
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


class BaseDetector(BaseModel):
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

    @overload
    def generate_detections(
        self,
        inputs: InputSample,
        features: FeatureMaps,
    ) -> Tuple[List[Boxes2D], Optional[List[InstanceMasks]]]:  # noqa: D102
        ...

    @overload
    def generate_detections(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[SamplingResult]]:
        ...

    @abc.abstractmethod
    def generate_detections(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: Optional[LabelInstances] = None,
    ) -> Union[
        Tuple[LossesType, Optional[SamplingResult]],
        Tuple[List[Boxes2D], Optional[List[InstanceMasks]]],
    ]:
        """Detector second stage (RoI Head).

        Return losses (empty if not training) and optionally detections.
        """
        raise NotImplementedError


class BaseTwoStageDetector(BaseModel):
    """Base class for two-stage detectors."""

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

    @overload
    def generate_proposals(
        self,
        inputs: InputSample,
        features: FeatureMaps,
    ) -> List[Boxes2D]:  # noqa: D102
        ...

    @overload
    def generate_proposals(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, List[Boxes2D]]:
        ...

    @abc.abstractmethod
    def generate_proposals(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: Optional[LabelInstances] = None,
    ) -> Union[Tuple[LossesType, List[Boxes2D]], List[Boxes2D]]:
        """Detector RPN stage.

        Return proposals per image and losses
        """
        raise NotImplementedError

    @overload
    def generate_detections(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        proposals: List[Boxes2D],
    ) -> Tuple[List[Boxes2D], Optional[List[InstanceMasks]]]:  # noqa: D102
        ...

    @overload
    def generate_detections(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        proposals: List[Boxes2D],
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[SamplingResult]]:
        ...

    @abc.abstractmethod
    def generate_detections(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        proposals: List[Boxes2D],
        targets: Optional[LabelInstances] = None,
    ) -> Union[
        Tuple[LossesType, Optional[SamplingResult]],
        Tuple[List[Boxes2D], Optional[List[InstanceMasks]]],
    ]:
        """Detector second stage (RoI Head).

        Return losses (empty if not training) and optionally detections.
        """
        raise NotImplementedError
