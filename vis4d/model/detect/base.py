"""Base class for Vis4D detectors."""

import abc
from typing import List, Optional, Tuple, Union, overload

from vis4d.common.bbox.samplers import SamplingResult
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
    ) -> Tuple[
        LossesType,
        Optional[Tuple[List[Boxes2D], Optional[List[InstanceMasks]]]],
    ]:
        ...

    @abc.abstractmethod
    def generate_detections(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: Optional[LabelInstances] = None,
    ) -> Union[
        Tuple[
            LossesType,
            Optional[Tuple[List[Boxes2D], Optional[List[InstanceMasks]]]],
        ],
        Tuple[List[Boxes2D], Optional[List[InstanceMasks]]],
    ]:
        """Detector second stage (RoI Head).

        Return losses and optionally detections and instance segmentations
        during training, and detections and instance segmentations during
        testing.
        """
        if targets is not None:
            return self._detections_train(inputs, features, targets)
        return self._detections_test(inputs, features)

    @abc.abstractmethod
    def _detections_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[
        LossesType,
        Optional[Tuple[List[Boxes2D], Optional[List[InstanceMasks]]]],
    ]:
        """Train stage detections generation."""
        raise NotImplementedError

    @abc.abstractmethod
    def _detections_test(
        self,
        inputs: InputSample,
        features: FeatureMaps,
    ) -> Tuple[List[Boxes2D], Optional[List[InstanceMasks]]]:
        """Test stage detections generation."""
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

    def generate_proposals(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: Optional[LabelInstances] = None,
    ) -> Union[Tuple[LossesType, List[Boxes2D]], List[Boxes2D]]:
        """Detector RPN stage.

        Return proposals per image and losses.
        """
        if targets is not None:
            return self._proposals_train(inputs, features, targets)
        return self._proposals_test(inputs, features)

    @abc.abstractmethod
    def _proposals_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, List[Boxes2D]]:
        """Train stage proposal generation."""
        raise NotImplementedError

    @abc.abstractmethod
    def _proposals_test(
        self,
        inputs: InputSample,
        features: FeatureMaps,
    ) -> List[Boxes2D]:
        """Test stage proposal generation."""
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
        if targets is not None:
            return self._detections_train(inputs, features, proposals, targets)
        return self._detections_test(inputs, features, proposals)

    @abc.abstractmethod
    def _detections_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        proposals: List[Boxes2D],
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[SamplingResult]]:
        """Train stage detections generation."""
        raise NotImplementedError

    @abc.abstractmethod
    def _detections_test(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        proposals: List[Boxes2D],
    ) -> Tuple[List[Boxes2D], Optional[List[InstanceMasks]]]:
        """Test stage detections generation."""
        raise NotImplementedError
