"""BBox coder base classes."""
import abc
from typing import List

import torch

from vis4d.struct_to_revise import Boxes2D, Boxes3D, Intrinsics


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


class BaseBoxCoder3D:
    """Base class for 3D box coders."""

    @abc.abstractmethod
    def encode(
        self,
        boxes: List[Boxes2D],
        targets: List[Boxes3D],
        intrinsics: Intrinsics,
    ) -> List[torch.Tensor]:
        """Encode deltas between boxes and targets given intrinsics."""
        raise NotImplementedError

    @abc.abstractmethod
    def decode(
        self,
        boxes: List[Boxes2D],
        box_deltas: List[torch.Tensor],
        intrinsics: Intrinsics,
    ) -> List[Boxes3D]:
        """Decode the predicted box_deltas according to given base boxes."""
        raise NotImplementedError