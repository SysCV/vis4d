"""Utils for samplers."""
from typing import Tuple

import torch

from openmt.common.bbox.matchers import MatchResult
from openmt.struct import Boxes2D


def prepare_target(
    num_pos: int,
    sampled_idcs: torch.Tensor,
    target: Boxes2D,
    match: MatchResult,
) -> Boxes2D:
    """Prepare target from sampled indices."""
    if len(target):
        sampled_target = target[match.assigned_gt_indices.long()[sampled_idcs]]
        assert sampled_target.class_ids is not None, "Targets have no class"
        sampled_target.class_ids[num_pos:] = -1
        if sampled_target.track_ids is not None:
            sampled_target.track_ids[num_pos:] = -1
    else:
        class_ids = torch.ones(len(sampled_idcs)).to(target.device) * -1
        track_ids = None
        if target.track_ids is not None:
            track_ids = class_ids.clone()
        sampled_target = Boxes2D(
            torch.zeros(len(sampled_idcs), 5).to(target.device), class_ids,
            track_ids
        )
    return sampled_target


def nonzero_tuple(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor]:  # pragma: no cover
    """A 'as_tuple=True' version of torch.nonzero to support torchscript.

    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if torch.jit.is_scripting():
        if tensor.dim() == 0:
            return tensor.unsqueeze(0).nonzero().unbind(1)  # type: ignore
        return tensor.nonzero().unbind(1)  # type: ignore
    return tensor.nonzero(as_tuple=True)  # type: ignore
