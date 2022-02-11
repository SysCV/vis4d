"""Utilities for augmentation."""
from typing import Tuple

import torch
from torch.distributions import Bernoulli


def sample_bernoulli(num: int, prob: float) -> torch.Tensor:
    """Sample from a Bernoulli distribution with given p."""
    curr_prob: torch.Tensor
    if prob == 1.0:
        curr_prob = torch.tensor([True] * num)
    elif prob == 0.0:
        curr_prob = torch.tensor([False] * num)
    else:
        curr_prob = Bernoulli(prob).sample((num,)).bool()
    return curr_prob


def sample_batched(num: int, prob: float, same: bool = False) -> torch.Tensor:
    """Sample num / 1 times from a Bernoulli distribution with given p."""
    if same:
        return sample_bernoulli(1, prob).repeat(num)
    return sample_bernoulli(num, prob)


def get_resize_shape(
    ori_wh: Tuple[int, int], new_wh: Tuple[int, int], keep_ratio: bool = True
) -> Tuple[int, int]:
    """Get shape for resize, considering keep_ratio."""
    w, h = ori_wh
    new_w, new_h = new_wh
    if keep_ratio:
        long_edge, short_edge = max(new_wh), min(new_wh)
        scale_factor = min(long_edge / max(h, w), short_edge / min(h, w))
        new_h = int(h * scale_factor + 0.5)
        new_w = int(w * scale_factor + 0.5)
    return new_w, new_h
