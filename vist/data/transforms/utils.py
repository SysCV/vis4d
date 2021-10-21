"""Utilities for augmentation."""
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
    """Sample num / 1 times from from a Bernoulli distribution with given p."""
    if same:
        return sample_bernoulli(1, prob).repeat(num)
    return sample_bernoulli(num, prob)
