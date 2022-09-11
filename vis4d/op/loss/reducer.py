"""Definitions of loss reducers.

Loss reducers are usually used as the last step in loss computation to average
or sum the loss maps from dense predictions or object detections.
"""

from typing import Callable

import torch

LossReducer = Callable[[torch.Tensor], torch.Tensor]


def identity_loss(loss: torch.Tensor) -> torch.Tensor:
    """Make no change to the loss."""
    return loss


def mean_loss(loss: torch.Tensor) -> torch.Tensor:
    """Average the loss tensor values to a single value.

    Args:
        loss (torch.Tensor): Input multi-dimentional tensor.

    Returns:
        torch.Tensor: Tensor containing a single loss value.
    """
    return loss.mean()


def sum_loss(loss: torch.Tensor) -> torch.Tensor:
    """Sum the loss tensor values to a single value.

    Args:
        loss (torch.Tensor): Input multi-dimentional tensor.

    Returns:
        torch.Tensor: Tensor containing a single loss value.
    """
    return loss.sum()


class SumWeightedLoss:
    """A loss reducer to calculated weighted sum loss."""

    def __init__(self, weight: torch.Tensor, avg_factor: float) -> None:
        """Initialize the loss reducer.

        Args:
            weight (torch.Tensor): Weights for each loss elements
            avg_factor (float): average factor for the weighted loss
        """
        self.weight = weight
        self.avg_factor = avg_factor

    def __call__(self, loss: torch.Tensor) -> torch.Tensor:
        """Weight the loss elements and take the sum with the average factor.

        Args:
            loss (torch.Tensor): input loss

        Returns:
            torch.Tensor: output loss
        """
        return (loss * self.weight).sum() / self.avg_factor
