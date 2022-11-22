"""Tracking model utils."""
from __future__ import annotations

import torch
from torch.nn import functional as F


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
