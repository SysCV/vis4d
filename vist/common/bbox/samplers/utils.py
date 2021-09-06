"""Utils for samplers."""
import torch

from vist.struct import Boxes2D


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
