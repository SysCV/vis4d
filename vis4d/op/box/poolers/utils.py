"""Utility functions for RoI poolers."""

from __future__ import annotations

import torch

from ..box2d import bbox_area


def assign_boxes_to_levels(
    box_lists: list[torch.Tensor],
    min_level: int,
    max_level: int,
    canonical_box_size: int,
    canonical_level: int,
) -> torch.Tensor:
    """Map each box to a feature map level index and return the assignment.

    Args:
        box_lists: List of Boxes
        min_level: Smallest feature map level index. The input is considered
            index 0, the output of stage 1 is index 1, and so.
        max_level: Largest feature map level index.
        canonical_box_size: A canonical box size in pixels (sqrt(box area)).
        canonical_level: The feature map level index on which a
            canonically-sized box should be placed.

    Returns:
        Tensor (M,), where M is the total number of boxes in the list. Each
        element is the feature map index, as an offset from min_level, for the
        corresponding box (so value i means the box is at self.min_level + i).
    """
    box_sizes = torch.sqrt(
        torch.cat([bbox_area(boxes) for boxes in box_lists])
    )
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8)
    )
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(
        level_assignments, min=min_level, max=max_level
    )
    return level_assignments.to(torch.int64) - min_level


def boxes_to_tensor(boxes: list[torch.Tensor]) -> torch.Tensor:
    """Convert all boxes into the tensor format used by ROI pooling ops.

    Args:
        boxes: List of Boxes

    Returns:
        A tensor of shape (M, 5), where M is the total number of boxes
        aggregated over all N batch images. The 5 columns are
        (batch index, x0, y0, x1, y1), where batch index is in [0, N).
    """

    def _fmt_box_list(box_tensor: torch.Tensor, batch_i: int) -> torch.Tensor:
        repeated_index = torch.full_like(
            box_tensor[:, :1],
            batch_i,
            dtype=box_tensor.dtype,
            device=box_tensor.device,
        )
        return torch.cat((repeated_index, box_tensor), dim=1)

    pooler_fmt_boxes = torch.cat(
        [_fmt_box_list(boxs[:, :4], i) for i, boxs in enumerate(boxes)],
        dim=0,
    )
    return pooler_fmt_boxes
