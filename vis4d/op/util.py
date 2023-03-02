"""Utilities for op."""
from __future__ import annotations

import copy

import torch
from torch import Tensor


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


def clone(module: torch.nn.Module, num_clones: int) -> torch.nn.ModuleList:
    """Deep copy a torch module N times.

    Args:
        module (torch.nn.Module): Module to clone.
        num_clones (int): Number of clones.

    Returns:
        torch.nn.ModuleList: List of cloned modules.
    """
    assert num_clones > 0, "Num_clones must be greater than 0"
    return torch.nn.ModuleList(
        [copy.deepcopy(module) for _ in range(num_clones)]
    )
