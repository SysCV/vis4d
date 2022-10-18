"""BBox coder base classes."""
import abc
from typing import List

import torch


class BoxEncoder2D:
    """Base class for 2D box coders."""

    @abc.abstractmethod
    def encode(
        self, boxes: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Encode deltas between boxes and targets."""
        raise NotImplementedError

    @abc.abstractmethod
    def decode(
        self, boxes: torch.Tensor, box_deltas: torch.Tensor
    ) -> torch.Tensor:
        """Decode the predicted box_deltas according to given base boxes."""
        raise NotImplementedError
