"""Utils for samplers."""
from typing import Dict, List, Union

import torch

from vist.struct import Boxes2D

from ..matchers import MatchResult


def prepare_target(
    sampled_idcs: torch.Tensor,
    target: Boxes2D,
    assigned_gt_indices: torch.Tensor,
) -> Boxes2D:
    """Prepare target from sampled indices."""
    if len(target):
        sampled_target = target[assigned_gt_indices.long()[sampled_idcs]]
    else:
        class_ids = torch.ones(len(sampled_idcs), device=target.device) * -1
        track_ids = None
        if target.track_ids is not None:
            track_ids = class_ids.clone()
        sampled_target = Boxes2D(
            torch.zeros(
                len(sampled_idcs), target.boxes.shape[1], device=target.device
            ),
            class_ids,
            track_ids,
        )
    return sampled_target


def add_to_result(
    result: Dict[str, Union[List[Boxes2D], List[torch.Tensor]]],
    sampled_idcs: torch.Tensor,
    boxes: Boxes2D,
    targets: Boxes2D,
    match: MatchResult,
) -> None:
    """Add fields required in SamplingResult to input dict."""
    result["sampled_boxes"] += [boxes[sampled_idcs]]
    result["sampled_targets"] += [
        prepare_target(sampled_idcs, targets, match.assigned_gt_indices)
    ]
    result["sampled_labels"] += [match.assigned_labels[sampled_idcs]]
    result["sampled_indices"] += [sampled_idcs]
    result["sampled_target_indices"] += [
        match.assigned_gt_indices[sampled_idcs]
    ]
