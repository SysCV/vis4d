"""Tracking model utils."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from vis4d.struct_to_revise import InputSample


def split_key_ref_inputs(
    inputs: list[InputSample],
) -> tuple[InputSample, list[InputSample]]:
    """Split key / ref frame inputs from List of InputSample."""
    key_ind = 0
    for i, s in enumerate(inputs):
        if s.metadata[0].attributes is not None and s.metadata[
            0
        ].attributes.get("keyframe", False):
            key_ind = i

    key_input = inputs.pop(key_ind)
    return key_input, inputs


def cosine_similarity(
    key_embeds: torch.Tensor,
    ref_embeds: torch.Tensor,
    normalize: bool = True,
    temperature: float = -1,
) -> torch.Tensor:
    """Calculate cosine similarity."""
    if normalize:
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)

    dists = torch.mm(key_embeds, ref_embeds.t())

    if temperature > 0:
        dists /= temperature  # pragma: no cover
    return dists
