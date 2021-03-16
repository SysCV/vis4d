"""Interface for openMT bounding box samplers."""

import abc
from typing import List

from openmt.core.registry import RegistryHolder
from openmt.label import Boxes2D

from ..matchers.base_matcher import MatchResult


class BaseSampler(metaclass=RegistryHolder):
    @abc.abstractmethod
    def sample(
        self, matching: MatchResult, boxes: List[Boxes2D], labels: List[int]
    ) -> List[Boxes2D]:
        """Sample bounding boxes according to their label."""
        raise NotImplementedError
