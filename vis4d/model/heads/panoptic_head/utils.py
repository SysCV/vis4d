"""Panoptic Head utils."""
from typing import Tuple

import torch

from vis4d.struct import InstanceMasks


def prune_instance_masks(
    ins_segm: InstanceMasks,
    thing_conf_thr: float = 0.5,
    overlap_thr: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prune instance masks."""
    # foreground mask
    foreground = torch.zeros(
        ins_segm.masks.shape[1:], dtype=torch.bool, device=ins_segm.device
    )

    # sort instance outputs by scores
    sorted_inds = ins_segm.score.argsort(descending=True)

    # add instances one-by-one, check for overlaps with existing ones
    inst_ids = []
    for inst_id in sorted_inds:
        mask = ins_segm.masks[inst_id]  # H, W
        score = ins_segm.score[inst_id].item()
        if score < thing_conf_thr:
            mask[mask > 0] = 0
            continue

        mask_area = mask.sum().item()
        if mask_area == 0:
            continue

        intersect = torch.logical_and(mask, foreground)
        intersect_area = intersect.sum().item()

        if intersect_area * 1.0 / mask_area > overlap_thr:
            mask[mask > 0] = 0
            continue

        if intersect_area > 0:
            ins_segm.masks[inst_id] = torch.logical_and(mask, ~foreground)
        foreground = torch.logical_or(mask, foreground)
        inst_ids.append(inst_id)
    return foreground, torch.tensor(inst_ids)
