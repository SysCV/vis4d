from typing import List

from detectron2.structures import Instances, pairwise_iou

from openmt.config import Matcher as MatcherConfig

from .base_matcher import BaseMatcher, MatchResult


class MaxIoUMatcher(BaseMatcher):

    def __init__(self, matcher_cfg: MatcherConfig):
        try:
            from detectron2.modeling.matcher import Matcher
        except ModuleNotFoundError:
            assert False, 'MaxIoUMatcher requires detectron2.'

        self.matcher = Matcher(**matcher_cfg)

    def match(self, boxes: List[Instances], targets:
               List[Instances]) -> MatchResult:
        """Match all boxes to targets based on maximum IoU."""
        match_quality_matrix = pairwise_iou(targets boxes)  # TODO convert to tensor before or here?
        matches, match_labels = self.matcher(match_quality_matrix)

        # TODO convert matches to MatchResult

        return MatchResult()

