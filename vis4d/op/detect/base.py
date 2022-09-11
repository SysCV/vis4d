"""Base class for Vis4D detectors."""

import abc
from typing import Dict, List, Optional, Tuple, Union, overload

from torch import nn

from vis4d.common_to_clean.bbox.samplers import SamplingResult
from vis4d.struct import (
    Boxes2D,
    InputSample,
    InstanceMasks,
    LabelInstances,
    LossesType,
    NamedTensors,
)


class BaseOneStageDetector(nn.Module):
    """Base single-stage detector class."""

    def __init__(
        self,
        image_channel_mode: str = "RGB",
        category_mapping: Optional[Dict[str, int]] = None,
        clip_bboxes_to_image: bool = True,
        resolve_overlap: bool = True,
    ):
        """Init."""
        super().__init__()
        self.clip_bboxes_to_image = clip_bboxes_to_image
        self.resolve_overlap = resolve_overlap
        self.category_mapping = category_mapping
        self.image_channel_mode = image_channel_mode

    @abc.abstractmethod
    def extract_features(self, inputs: InputSample) -> NamedTensors:
        """Detector feature extraction stage.

        Return backbone output features
        """
        raise NotImplementedError

    @overload
    def generate_detections(
        self, inputs: InputSample, features: NamedTensors
    ) -> List[Boxes2D]:  # noqa: D102
        ...

    @overload
    def generate_detections(
        self,
        inputs: InputSample,
        features: NamedTensors,
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[List[Boxes2D]]]:
        ...

    @abc.abstractmethod
    def generate_detections(
        self,
        inputs: InputSample,
        features: NamedTensors,
        targets: Optional[LabelInstances] = None,
    ) -> Union[
        Tuple[LossesType, Optional[List[Boxes2D]]], List[Boxes2D]
    ]:  # pragma: no cover
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
        features: NamedTensors,
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[List[Boxes2D]]]:
        """Train stage detections generation."""
        raise NotImplementedError

    @abc.abstractmethod
    def _detections_test(
        self, inputs: InputSample, features: NamedTensors
    ) -> List[Boxes2D]:
        """Test stage detections generation."""
        raise NotImplementedError


class BaseTwoStageDetector(nn.Module):
    """Base class for two-stage detectors."""

    def __init__(
        self,
        image_channel_mode: str = "RGB",
        category_mapping: Optional[Dict[str, int]] = None,
        clip_bboxes_to_image: bool = True,
        resolve_overlap: bool = True,
    ):
        """Init."""
        super().__init__()
        self.clip_bboxes_to_image = clip_bboxes_to_image
        self.resolve_overlap = resolve_overlap
        self.category_mapping = category_mapping
        self.image_channel_mode = image_channel_mode

    @abc.abstractmethod
    def extract_features(self, inputs: InputSample) -> NamedTensors:
        """Detector feature extraction stage.

        Return backbone output features
        """
        raise NotImplementedError

    @overload
    def generate_proposals(
        self, inputs: InputSample, features: NamedTensors
    ) -> List[Boxes2D]:  # noqa: D102
        ...

    @overload
    def generate_proposals(
        self,
        inputs: InputSample,
        features: NamedTensors,
        targets: LabelInstances,
    ) -> Tuple[LossesType, List[Boxes2D]]:
        ...

    def generate_proposals(
        self,
        inputs: InputSample,
        features: NamedTensors,
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
        features: NamedTensors,
        targets: LabelInstances,
    ) -> Tuple[LossesType, List[Boxes2D]]:
        """Train stage proposal generation."""
        raise NotImplementedError

    @abc.abstractmethod
    def _proposals_test(
        self, inputs: InputSample, features: NamedTensors
    ) -> List[Boxes2D]:
        """Test stage proposal generation."""
        raise NotImplementedError

    @overload
    def generate_detections(
        self,
        inputs: InputSample,
        features: NamedTensors,
        proposals: List[Boxes2D],
    ) -> Tuple[List[Boxes2D], Optional[List[InstanceMasks]]]:  # noqa: D102
        ...

    @overload
    def generate_detections(
        self,
        inputs: InputSample,
        features: NamedTensors,
        proposals: List[Boxes2D],
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[SamplingResult]]:
        ...

    def generate_detections(
        self,
        inputs: InputSample,
        features: NamedTensors,
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
        features: NamedTensors,
        proposals: List[Boxes2D],
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[SamplingResult]]:
        """Train stage detections generation."""
        raise NotImplementedError

    @abc.abstractmethod
    def _detections_test(
        self,
        inputs: InputSample,
        features: NamedTensors,
        proposals: List[Boxes2D],
    ) -> Tuple[List[Boxes2D], Optional[List[InstanceMasks]]]:
        """Test stage detections generation."""
        raise NotImplementedError


BaseDetector = Union[BaseOneStageDetector, BaseTwoStageDetector]
