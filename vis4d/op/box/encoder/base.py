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


class BoxEncoder3D:
    """Base class for 3D box coders."""

    # @abc.abstractmethod
    # def encode(
    #     self,
    #     boxes_3d: Tensor,
    #     targets: Tensor,
    #     intrinsics: Tensor,
    # ) -> Tensor:
    #     """Encode deltas between 3D boxes and targets given intrinsics."""
    #     raise NotImplementedError

    @abc.abstractmethod
    def decode(
        self,
        boxes_2d: Tensor,
        boxes_deltas: Tensor,
        intrinsics: Tensor,
    ) -> Tensor:
        """Decode the predicted box_deltas according to given base boxes."""
        raise NotImplementedError
