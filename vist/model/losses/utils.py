"""Loss utility functions."""
from typing import Optional

import torch
from torch.nn._reduction import get_enum


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduce loss based on pytorch reduction logic."""
    value = get_enum(reduction)
    if value == 0:
        return loss
    if value == 1:
        return loss.mean()
    if value == 2:
        return loss.sum()
    raise ValueError(f"Invalid reduction: {reduction}.")


def weight_reduce_loss(
    loss: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    avg_factor: Optional[float] = None,
) -> torch.Tensor:
    """Apply element-wise weight and reduce loss."""
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = _reduce(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":  # pragma: no cover
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    avg_factor: Optional[float] = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth L1 loss."""
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(
        diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta
    )
    if weight is not None:
        weight = weight.float()
    return weight_reduce_loss(loss, weight, reduction, avg_factor)


def l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    avg_factor: Optional[float] = None,
) -> torch.Tensor:
    """L2 loss."""
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target) ** 2
    if weight is not None:
        weight = weight.float()
    return weight_reduce_loss(loss, weight, reduction, avg_factor)
