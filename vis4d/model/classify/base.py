"""Base class for Vis4D classifiers."""

import abc
from typing import List, Optional, Tuple, Union, overload

from vis4d.struct import (
    FeatureMaps,
    ImageTags,
    InputSample,
    LabelInstances,
    LossesType,
)

from ..base import BaseModel


class BaseClassifier(BaseModel):
    """Base classifier class."""

    @abc.abstractmethod
    def extract_features(self, inputs: InputSample) -> FeatureMaps:
        """Classifier feature extraction stage.

        Return backbone output features
        """
        raise NotImplementedError

    @overload
    def generate_classifications(
        self, inputs: InputSample, features: FeatureMaps
    ) -> List[ImageTags]:  # noqa: D102
        ...

    @overload
    def generate_classifications(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[List[ImageTags]]]:
        ...

    @abc.abstractmethod
    def generate_classifications(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: Optional[LabelInstances] = None,
    ) -> Union[
        Tuple[LossesType, Optional[List[ImageTags]]], List[ImageTags]
    ]:  # pragma: no cover
        """Classifications second stage (Cls Head).

        Return losses and optionally classifications during training, and
        classifications during testing.
        """
        if targets is not None:
            return self._classifications_train(inputs, features, targets)
        return self._classifications_test(inputs, features)

    @abc.abstractmethod
    def _classifications_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[List[ImageTags]]]:
        """Train stage classifications generation."""
        raise NotImplementedError

    @abc.abstractmethod
    def _classifications_test(
        self, inputs: InputSample, features: FeatureMaps
    ) -> List[ImageTags]:
        """Test stage classifications generation."""
        raise NotImplementedError
