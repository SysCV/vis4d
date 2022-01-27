"""Base class for meta architectures."""

import abc
from typing import Any, Optional

import torch

from vis4d.common import Vis4DModule


class BaseLoss(Vis4DModule[torch.Tensor, torch.Tensor]):
    """Base loss class."""

    def __init__(
        self, reduction: str = "mean", loss_weight: Optional[float] = 1.0
    ) -> None:
        """Init."""
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    @abc.abstractmethod
    def __call__(  # type: ignore
        self, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Loss function implementation.

        Returns the reduced loss (scalar).
        """
        raise NotImplementedError
