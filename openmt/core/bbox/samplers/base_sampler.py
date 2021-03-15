"""Interface for openMT bounding box samplers."""

import abc
from typing import List

from detectron2.structures import (  # TODO either integrate class or write own
    Instances,
)


class BaseSampler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(
        self, matching: MatchResult, boxes: List[Instances], labels: List[int]
    ) -> List[Instances]:
        """Sample bounding boxes according to their label."""
        raise NotImplementedError
