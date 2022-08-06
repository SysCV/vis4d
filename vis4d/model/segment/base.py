"""Base class for Vis4D segmentors."""

import abc
from typing import Dict, List, Optional, Tuple, Union, overload

from torch import nn

from vis4d.struct import (
    NamedTensors,
    InputSample,
    LabelInstances,
    LossesType,
    SemanticMasks,
)


class BaseSegmentor(nn.Module):
    """Base segmentor class."""

    def __init__(
        self,
        image_channel_mode: str = "RGB",
        category_mapping: Optional[Dict[str, int]] = None,
    ):
        """Init."""
        super().__init__()
        self.category_mapping = category_mapping
        self.image_channel_mode = image_channel_mode

    @abc.abstractmethod
    def extract_features(self, inputs: InputSample) -> NamedTensors:
        """Segmentor feature extraction stage.

        Return backbone output features.
        """
        raise NotImplementedError

    @overload
    def generate_segmentations(
        self, inputs: InputSample, features: NamedTensors
    ) -> List[SemanticMasks]:
        ...

    @overload
    def generate_segmentations(
        self,
        inputs: InputSample,
        features: NamedTensors,
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[List[SemanticMasks]]]:
        ...

    def generate_segmentations(
        self,
        inputs: InputSample,
        features: NamedTensors,
        targets: Optional[LabelInstances] = None,
    ) -> Union[
        Tuple[
            LossesType,
            Optional[List[SemanticMasks]],
        ],
        List[SemanticMasks],
    ]:
        """Segmentor decode stage.

        Return losses and optionally segmentations during training, and
        segmentations during testing.
        """
        if targets is not None:
            return self._segmentations_train(inputs, features, targets)
        return self._segmentations_test(inputs, features)

    @abc.abstractmethod
    def _segmentations_train(
        self,
        inputs: InputSample,
        features: NamedTensors,
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[List[SemanticMasks]]]:
        """Train stage segmentations generation."""
        raise NotImplementedError

    @abc.abstractmethod
    def _segmentations_test(
        self, inputs: InputSample, features: NamedTensors
    ) -> List[SemanticMasks]:
        """Test stage segmentations generation."""
        raise NotImplementedError
