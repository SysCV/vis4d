from typing import List

from detectron2.modeling.matcher import Matcher
from detectron2.structures import Boxes, pairwise_iou

from openmt.config import Matcher as MatcherConfig
from openmt.label import Boxes2D

from .base_matcher import BaseMatcher, MatchResult


class MaxIoUMatcher(BaseMatcher):
    def __init__(self, matcher_cfg: MatcherConfig):
        """Init."""
        self.matcher = Matcher(**matcher_cfg.__dict__)

    def match(
        self, boxes: List[Boxes2D], targets: List[Boxes2D]
    ) -> List[MatchResult]:
        """Match all boxes to targets based on maximum IoU."""

        result = []
        for b, t in zip(boxes, targets):
            match_quality_matrix = pairwise_iou(
                Boxes(b.data[:, :4]), Boxes(t.data[:, :4])
            )
            matches, match_labels = self.matcher(match_quality_matrix)

            # TODO convert matches to MatchResult
            result.append(MatchResult(matches, match_labels))

        return result
