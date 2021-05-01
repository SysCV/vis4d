"""Tracking model utils."""
import random
from enum import Enum
from typing import List, Tuple

import torch
from torch.nn import functional as F


class KeyFrameSelection(str, Enum):
    """Enum for key frame selection strategy.

    random: Keyframe is randomly selected from the input frames.
    first: Keyframe == first frame.
    last:  Keyframe == last frame.
    """

    RANDOM = "random"
    FIRST = "first"
    LAST = "last"


def select_keyframe(
    sequence_length: int, strategy: str = "random"
) -> Tuple[int, List[int]]:
    """Keyframe selection.

    Strategies:
    - Random
    - First frame
    - Last frame
    """
    if strategy == "random":
        key_index = random.randint(0, sequence_length - 1)
    elif strategy == "first":
        key_index = 0
    elif strategy == "last":
        key_index = sequence_length - 1
    else:
        raise NotImplementedError(
            f"Keyframe selection strategy {strategy} not implemented"
        )

    ref_indices = list(range(sequence_length))
    ref_indices.remove(key_index)

    return key_index, ref_indices


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
