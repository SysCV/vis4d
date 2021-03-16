"""Matchers."""
import abc
from typing import List

from pydantic import BaseModel

from openmt.core.registry import RegistryHolder
from openmt.label import Boxes2D


class MatchResult(BaseModel):
    gt_mapping: List[int]


class BaseMatcher(metaclass=RegistryHolder):
    @abc.abstractmethod
    def match(
        self, boxes: List[Boxes2D], targets: List[Boxes2D]
    ) -> MatchResult:
        """Match bounding boxes according to their label."""

        raise NotImplementedError
