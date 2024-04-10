"""Matching calculation utils."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def calc_bisoftmax_affinity(
    detection_embeddings: torch.Tensor,
    track_embeddings: torch.Tensor,
    detection_class_ids: torch.Tensor | None = None,
    track_class_ids: torch.Tensor | None = None,
    with_categories: bool = False,
) -> torch.Tensor:
    """Calculate affinity matrix using bisoftmax metric."""
    feats = torch.mm(detection_embeddings, track_embeddings.t())
    d2t_scores = feats.softmax(dim=1)
    t2d_scores = feats.softmax(dim=0)
    similarity_scores = (d2t_scores + t2d_scores) / 2

    if with_categories:
        assert (
            detection_class_ids is not None and track_class_ids is not None
        ), "Please provide class ids if with_categories=True!"
        cat_same = detection_class_ids.view(-1, 1) == track_class_ids.view(
            1, -1
        )
        similarity_scores *= cat_same.float()
    return similarity_scores


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
