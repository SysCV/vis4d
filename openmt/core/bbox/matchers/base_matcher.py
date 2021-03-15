"""Matchers."""
import abc
from typing import List

from detectron2.structures import Instances  # TODO replace
from pydantic import BaseModel

from openmt.config import Matcher as MatcherConfig


class MatchResult(BaseModel):
    gt_mapping: List[int]


class BaseMatcher(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def match(
        self, boxes: List[Instances], targets: List[Instances]
    ) -> MatchResult:
        """Match bounding boxes according to their label."""
        raise NotImplementedError
