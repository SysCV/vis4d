from typing import List

from detectron2.modeling.matcher import Matcher
from detectron2.structures import Boxes, pairwise_iou

from openmt.config import Matcher as MatcherConfig
from openmt.structures import Boxes2D

from .base_matcher import BaseMatcher, MatchResult


class MaxIoUMatcher(BaseMatcher):
    def __init__(self, matcher_cfg: MatcherConfig):
        """Init."""
        cfg_dict = matcher_cfg.__dict__
        del cfg_dict["type"]
        self.matcher = Matcher(**cfg_dict)

    def match(
        self, boxes: List[Boxes2D], targets: List[Boxes2D]
    ) -> List[MatchResult]:
        """Match all boxes to targets based on maximum IoU."""

        result = []
        for b, t in zip(boxes, targets):
            # M x N matrix, where M = num gt, N = num proposals
            match_quality_matrix = pairwise_iou(t.gt_boxes, b.proposal_boxes)
            # Boxes(b.data[:, :4]), Boxes(t.data[:, :4]) TODO
            # )

            # matches N x 1 = index of assigned gt i.e.  range [0, M)
            # match_labels N x 1 where 0 = negative, -1 = ignore, 1 = positive
            matches, match_labels = self.matcher(match_quality_matrix)
            result.append(
                MatchResult(
                    **dict(
                        assigned_gt_indices=matches,
                        assigned_labels=match_labels,
                    )
                )
            )

        return result
