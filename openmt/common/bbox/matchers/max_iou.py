"""Match predictions and targets according to maximum 2D IoU."""
from typing import List

import torch
from detectron2.modeling.matcher import Matcher as D2Matcher

from openmt.common.bbox.utils import compute_iou
from openmt.struct import Boxes2D

from .base import BaseMatcher, MatcherConfig, MatchResult


class MaxIoUMatcherConfig(MatcherConfig):
    """MaxIoUMatcher config."""

    thresholds: List[float]
    labels: List[int]
    allow_low_quality_matches: bool


class MaxIoUMatcher(BaseMatcher):
    """MaxIoUMatcher class. Based on Matcher in detectron2."""

    def __init__(self, cfg: MatcherConfig):
        """Init."""
        super().__init__()
        self.cfg = MaxIoUMatcherConfig(**cfg.dict())

        self.matcher = D2Matcher(
            thresholds=self.cfg.thresholds,
            labels=self.cfg.labels,
            allow_low_quality_matches=self.cfg.allow_low_quality_matches,
        )

    def match(
        self, boxes: List[Boxes2D], targets: List[Boxes2D]
    ) -> List[MatchResult]:
        """Match all boxes to targets based on maximum IoU."""
        result = []
        for b, t in zip(boxes, targets):
            if len(t) == 0:
                matches = torch.zeros(len(b)).to(b.device)
                match_labels = torch.zeros(len(b)).to(b.device)
                match_iou = torch.zeros(len(b)).to(b.device)
            else:
                # M x N matrix, where M = num gt, N = num proposals
                match_quality_matrix = compute_iou(t, b)

                # matches N x 1 = index of assigned gt i.e.  range [0, M)
                # match_labels N x 1, 0 = negative, -1 = ignore, 1 = positive
                matches, match_labels = self.matcher(match_quality_matrix)
                match_iou = match_quality_matrix[
                    matches, torch.arange(0, len(b)).to(b.device)
                ]

            result.append(
                MatchResult(
                    assigned_gt_indices=matches,
                    assigned_labels=match_labels,
                    assigned_gt_iou=match_iou,
                )
            )
        return result
