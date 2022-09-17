"""Common loss functions."""
import torch

from vis4d.op.loss.reducer import LossReducer, identity_loss


def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reducer: LossReducer = identity_loss,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth L1 loss."""
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(
        diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta
    )
    return reducer(loss)


def l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reducer: LossReducer = identity_loss,
) -> torch.Tensor:
    """L1 loss."""
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return reducer(loss)


def l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reducer: LossReducer = identity_loss,
) -> torch.Tensor:
    """L2 loss."""
    assert pred.size() == target.size() and target.numel() > 0
    loss = (pred - target) ** 2
    return reducer(loss)
