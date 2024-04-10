"""Matchers."""

import abc
from typing import NamedTuple

import torch
from torch import nn


class MatchResult(NamedTuple):
    """Match result class. Stores expected result tensors.

    assigned_gt_indices: torch.Tensor - Tensor of [0, M) where M = num gt
    assigned_gt_iou: torch.Tensor  - Tensor with IoU to assigned GT
    assigned_labels: torch.Tensor  - Tensor of {0, -1, 1} = {neg, ignore, pos}
    """

    assigned_gt_indices: torch.Tensor
    assigned_gt_iou: torch.Tensor
    assigned_labels: torch.Tensor


class Matcher(nn.Module):
    """Base class for box / target matchers."""

    @abc.abstractmethod
    def forward(
        self, boxes: torch.Tensor, targets: torch.Tensor
    ) -> MatchResult:
        """Match bounding boxes according to their struct."""
        raise NotImplementedError

    def __call__(
        self, boxes: torch.Tensor, targets: torch.Tensor
    ) -> MatchResult:
        """Type declaration for forward."""
        return self._call_impl(boxes, targets)
