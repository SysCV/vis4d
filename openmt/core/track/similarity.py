"""Implementation of embedding similarity measures for tracking."""

import torch
import torch.nn.functional as F


def cosine_similarity(key_embeds, ref_embeds, normalize=True, temperature=-1):
    """Calculate cosine similarity."""

    if normalize:
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)

    dists = torch.mm(key_embeds, ref_embeds.t())

    if temperature > 0:
        dists /= temperature
    return dists
