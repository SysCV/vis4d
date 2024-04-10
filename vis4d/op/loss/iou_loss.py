"""Embedding distance loss."""

from __future__ import annotations

import torch

from vis4d.op.box.box2d import bbox_iou_aligned

from .base import Loss
from .reducer import LossReducer, identity_loss


def iou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reducer: LossReducer = identity_loss,
    mode: str = "log",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute IoU loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Target bboxes.
        reducer (LossReducer): Reducer to reduce the loss value. Defaults to
            identy_loss, which is no reduction.
        mode (str, optional): Mode to calculate the loss. Defaults to "log".
        eps (float, optional): Epsilon value to avoid division by zero.

    Returns:
        torch.Tensor : The reduced IoU loss.
    """
    assert mode in {
        "linear",
        "square",
        "log",
    }, f"Invalid mode {mode}. Must be one of 'linear', 'square', 'log'."
    ious = bbox_iou_aligned(pred, target).clamp(min=eps)
    if mode == "linear":
        loss = 1 - ious
    elif mode == "square":
        loss = 1 - ious**2
    else:
        loss = -ious.log()
    return reducer(loss)


class IoULoss(Loss):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated depending on the mode:
        - linear: 1 - IoU
        - square: 1 - IoU^2
        - log: -log(IoU)

    Args:
        reducer (LossReducer): Reducer to reduce the loss value. Defaults to
            identy_loss, which is no reduction.
        mode (str, optional): Mode to calculate the loss. Defaults to "log".
        eps (float, optional): Epsilon value to avoid division by zero.
    """

    def __init__(
        self,
        reducer: LossReducer = identity_loss,
        mode: str = "log",
        eps: float = 1e-6,
    ):
        """Creates an instance of the class."""
        super().__init__(reducer)
        self.mode = mode
        self.eps = eps
        assert mode in {
            "linear",
            "square",
            "log",
        }, f"Invalid mode {mode}. Must be one of 'linear', 'square', 'log'."

    def forward(  # pylint: disable=arguments-differ
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted bboxes.
            target (torch.Tensor): Target bboxes.

        Returns:
            torch.Tensor: The reduced IoU loss.
        """
        return iou_loss(
            pred, target, reducer=self.reducer, mode=self.mode, eps=self.eps
        )
