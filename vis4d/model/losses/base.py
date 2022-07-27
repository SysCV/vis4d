"""Base class for meta architectures."""

import abc
from typing import Optional

from torch import nn


class BaseLoss(nn.Module, abc.ABC):
    """Base loss class."""

    def __init__(
        self, reduction: str = "mean", loss_weight: Optional[float] = 1.0
    ) -> None:
        """Init."""
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
