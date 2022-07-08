"""Base class for Vis4D segmentors."""

import abc
from typing import List, Optional, Tuple, Union, overload

from vis4d.struct import FeatureMaps, InputSample, Losses, SemanticMasks

from ..base import BaseModel


class BaseSegmentor(BaseModel):
    """Base segmentor class."""

    @abc.abstractmethod
    def extract_features(self, inputs: InputSample) -> FeatureMaps:
        """Segmentor feature extraction stage.

        Return backbone output features.
        """
        raise NotImplementedError

    @overload
    def generate_segmentations(
        self, inputs: InputSample, features: FeatureMaps
    ) -> List[SemanticMasks]:
        ...

    @overload
    def generate_segmentations(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets,
    ) -> Tuple[Losses, Optional[List[SemanticMasks]]]:
        ...

    def generate_segmentations(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets=None,
    ) -> Union[
        Tuple[
            Losses,
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
        targets,
    ) -> Tuple[Losses, Optional[List[SemanticMasks]]]:
        """Train stage segmentations generation."""
        raise NotImplementedError

    @abc.abstractmethod
    def _segmentations_test(
        self, inputs: InputSample, features: FeatureMaps
    ) -> List[SemanticMasks]:
        """Test stage segmentations generation."""
        raise NotImplementedError
