"""Utilities for op."""
from __future__ import annotations

import re

import torch
from torch import Tensor, nn

from vis4d.engine.ckpt import load_checkpoint

BDD100K_MODEL_PREFIX = "https://dl.cv.ethz.ch/bdd100k/"
MM_MODEL_MAP = {
    "mmdet://": "https://download.openmmlab.com/mmdetection/v2.0/",
    "mmseg://": "https://download.openmmlab.com/mmsegmentation/v0.5/",
}
MM_CFG_MAP = {
    "mmdet://": "syscv/mmdetection/master/configs/",
    "mmseg://": "open-mmlab/mmsegmentation/master/configs/",
}
MM_ZIP_MAP = {
    "mmdet://": "mmdetection-master/configs/",
    "mmseg://": "mmsegmentation-master/configs/",
}


def unmap(data: Tensor, count: int, inds: Tensor, fill: int = 0) -> Tensor:
    """Unmap a subset of data back to the original data (of size count).

    Args:
        data (Tensor): Subset of the original data.
        count (int): Length of the original data.
        inds (Tensor): Indices of the subset entries in the original set.
        fill (int, optional): Fill value for other entries. Defaults to 0.

    Returns:
        Tensor: Tensor sized like original data that contains the subset.
    """
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


# TODO this needs to be moved to vis4d.engine.ckpt if still needed
def load_model_checkpoint(
    model: nn.Module,
    weights: str,
    strict: bool = False,
    rev_keys: None | list[tuple[str, str]] = None,
) -> None:
    """Load MM model checkpoint."""
    if rev_keys is None:  # pragma: no cover
        rev_keys = [(r"^module\.", "")]
    if re.compile(r"^mm(det|seg)://").search(weights):
        pre = weights[:8]
        weights = MM_MODEL_MAP[pre] + weights.split(pre)[-1]
        load_checkpoint(model, weights, strict=strict, revise_keys=rev_keys)
    elif weights.startswith("bdd100k://"):
        weights = BDD100K_MODEL_PREFIX + weights.split("bdd100k://")[-1]
        load_checkpoint(model, weights, strict=strict, revise_keys=rev_keys)
    else:  # pragma: no cover
        load_checkpoint(model, weights, strict=strict, revise_keys=rev_keys)
