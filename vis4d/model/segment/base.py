"""Base class for Vis4D segmentors."""

import abc
from typing import List, Optional, Tuple

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
    ) -> Tuple[LossesType, Optional[List[SemanticMasks]]]:
        """Segmentor decode stage.

        Return losses (empty if not training) and optionally segmentations.
        """
        raise NotImplementedError
