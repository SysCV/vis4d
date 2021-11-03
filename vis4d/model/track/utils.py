"""Tracking model utils."""
from typing import List, Tuple

import torch
from torch.nn import functional as F

from vis4d.struct import InputSample


def split_key_ref_inputs(
    batched_input_samples: List[List[InputSample]],
) -> Tuple[List[InputSample], List[List[InputSample]]]:
    """Split key / ref frame inputs from batched List of InputSample."""
    key_indices = []  # type: List[int]
    ref_indices = []  # type: List[List[int]]
    for input_samples in batched_input_samples:
        curr_ref_indices = list(range(0, len(input_samples)))
        for i, sample in enumerate(input_samples):
            if sample.metadata[0].attributes is not None and sample.metadata[
                0
            ].attributes.get("keyframe", False):
                key_indices.append(curr_ref_indices.pop(i))
                ref_indices.append(curr_ref_indices)
                break

    key_inputs = [
        inputs[key_index]
        for inputs, key_index in zip(batched_input_samples, key_indices)
    ]
    ref_inputs = [
        [inputs[i] for i in curr_ref_indices]
        for inputs, curr_ref_indices in zip(batched_input_samples, ref_indices)
    ]
    return key_inputs, ref_inputs


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
