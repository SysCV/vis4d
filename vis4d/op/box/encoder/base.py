"""BBox coder base classes."""
import abc

from torch import Tensor


class BoxEncoder2D:
    """Base class for 2D box coders."""

    @abc.abstractmethod
    def encode(self, boxes: Tensor, targets: Tensor) -> Tensor:
        """Encode deltas between boxes and targets."""
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, boxes: Tensor, box_deltas: Tensor) -> Tensor:
        """Decode the predicted box_deltas according to given base boxes."""
        raise NotImplementedError
