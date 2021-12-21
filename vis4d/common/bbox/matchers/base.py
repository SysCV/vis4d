"""Matchers."""
import abc
from typing import List, NamedTuple

import torch

from vis4d.common import Vis4DModule
from vis4d.struct import Boxes2D


class MatchResult(NamedTuple):
    """Match result class. Stores expected result tensors.

    assigned_gt_indices: torch.Tensor - Tensor of [0, M) where M = num gt
    assigned_gt_iou: torch.Tensor  - Tensor with IoU to assigned GT
    assigned_labels: torch.Tensor  - Tensor of {0, -1, 1} = {neg, ignore, pos}
    """

    assigned_gt_indices: torch.Tensor
    assigned_gt_iou: torch.Tensor
    assigned_labels: torch.Tensor


class BaseMatcher(Vis4DModule[List[MatchResult], List[MatchResult]]):
    """Base class for box / target matchers."""

    @abc.abstractmethod
    def __call__(  # type: ignore
        self, boxes: List[Boxes2D], targets: List[Boxes2D]
    ) -> List[MatchResult]:
        """Match bounding boxes according to their struct."""
        raise NotImplementedError
