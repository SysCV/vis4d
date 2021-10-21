"""Utilities for augmentation."""
import torch
from torch.distributions import Bernoulli


def sample_bernoulli(prob: float) -> torch.Tensor:
    """Sample from a Bernoulli distribution with given p."""
    curr_prob: torch.Tensor
    if prob == 1.0:
        curr_prob = torch.tensor([True])
    elif prob == 0.0:
        curr_prob = torch.tensor([False])
    else:
        curr_prob = Bernoulli(prob).sample((1,)).bool()
    return curr_prob
