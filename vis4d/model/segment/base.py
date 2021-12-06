"""Base class for Vis4D segmentors."""

import abc
from typing import List, Optional, Tuple, Union

from vis4d.common.registry import RegistryHolder
from vis4d.struct import (
    FeatureMaps,
    InputSample,
    LabelInstances,
    LossesType,
    SemanticMasks,
)

from ..base import BaseModel


class BaseSegmentor(BaseModel, metaclass=RegistryHolder):
    """Base segmentor class."""

    @abc.abstractmethod
    def extract_features(self, inputs: InputSample) -> FeatureMaps:
        """Segmentor feature extraction stage.

        Return backbone output features.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_segmentations(
        self,
        inputs: InputSample,
        features: FeatureMaps,
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
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[List[SemanticMasks]]]:
        """Train stage segmentations generation."""
        raise NotImplementedError

    @abc.abstractmethod
    def _segmentations_test(
        self,
        inputs: InputSample,
        features: FeatureMaps,
    ) -> List[SemanticMasks]:
        """Test stage segmentations generation."""
        raise NotImplementedError
