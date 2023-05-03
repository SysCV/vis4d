"""Common loss functions."""
import torch
import torch.nn.functional as F

from vis4d.op.loss.reducer import LossReducer, identity_loss


def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reducer: LossReducer = identity_loss,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth L1 loss.

    L1 loss that uses a squared term if the absolute element-wise error
    falls below beta.

    Args:
        pred (torch.Tensor): Model predictions
        target (torch.Tensor): Ground truth value
        reducer (LossReducer): Reducer to reduce the loss value. Defaults to
            identy_loss, which is no reduction.
        beta (float): Specifies the threshold at which to change between L1
            and L2 loss. The value must be non-negative. Default: 1.0

    Returns:
        torch.Tensor : The reduced smooth l1 loss:
            |pred - target| - 0.5*beta if |pred - target| < 0.5*beta
            (pred - target)^2 * 0.5/beta else
    """
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
    """L1 loss.

    Args:
        pred (torch.Tensor): Model predictions
        target (torch.Tensor): Ground truth value
        reducer (LossReducer): Reducer to reduce the loss value. Defaults to
            identy_loss, which is no reduction.

    Returns:
        torch.Tensor : The reduced L1 loss (reduce(|pred - target|))
    """
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return reducer(loss)


def l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reducer: LossReducer = identity_loss,
) -> torch.Tensor:
    """L2 loss.

    Args:
        pred (torch.Tensor): Model predictions
        target (torch.Tensor): Ground truth value
        reducer (LossReducer): Reducer to reduce the loss value. Defaults to
            identy_loss, which is no reduction.

    Returns:
        torch.Tensor : The reduced L2 loss (reduce((pred - target)**2))
    """
    assert pred.size() == target.size() and target.numel() > 0
    loss = (pred - target) ** 2
    return reducer(loss)


def rotation_loss(
    pred: torch.Tensor,
    target_bin: torch.Tensor,
    target_res: torch.Tensor,
    num_bins: int,
    reducer: LossReducer = identity_loss,
) -> torch.Tensor:
    """Rotation loss.

    Consists of bin-based classification loss and residual-based regression
    loss.

    Args:
        pred (torch.Tensor): Prediction shape [B, num_bins * 3]
        target_bin (torch.Tensor): Target bins shape [B, num_bin]
        target_res (torch.Tensor): Target residual shape [B, num_bin]
        num_bins (int): Number of bins
        reducer (LossReducer, optional): Loss Reducer.
            Defaults to identity_loss.

    Returns:
        torch.Tensor: The reduced loss value
    """
    loss_bins = (
        F.binary_cross_entropy_with_logits(
            pred[:, :num_bins], target_bin, reduction="none"
        )
        .mean(dim=0)
        .sum()
    )

    loss_res = torch.zeros_like(loss_bins)
    for i in range(num_bins):
        bin_mask = target_bin[:, i] == 1
        res_idx = num_bins + 2 * i
        if bin_mask.any():
            loss_sin = smooth_l1_loss(
                pred[bin_mask, res_idx],
                torch.sin(target_res[bin_mask, i]),
                reducer=reducer,
            )
            loss_cos = smooth_l1_loss(
                pred[bin_mask, res_idx + 1],
                torch.cos(target_res[bin_mask, i]),
                reducer=reducer,
            )
            loss_res += loss_sin + loss_cos

    return loss_bins + loss_res


def inverse_sigmoid(inputs: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse sigmoid function.

    inv_sigmoid(x) := log(x / (1 - x))

    Args:
        inputs (torch.Tensor): Input tensor
        eps (float, optional): Epsilon value. Defaults to 1e-5.

    Returns:
        torch.Tensor: Inverse sigmoid value
    """
    inputs = inputs.clamp(min=0, max=1)
    x1 = inputs.clamp(min=eps)
    x2 = (1 - inputs).clamp(min=eps)
    return torch.log(x1 / x2)
