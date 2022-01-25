"""BBox coder base classes."""
import abc
from typing import List, Union

import torch

from vis4d.common.module import Vis4DModule
from vis4d.struct import Boxes2D, Boxes3D, Intrinsics


class BaseBoxCoder2D(Vis4DModule[List[torch.Tensor], List[Boxes2D]]):
    """Base class for 2D box coders."""

    @abc.abstractmethod
    def encode(
        self, boxes: List[Boxes2D], targets: List[Boxes2D]
    ) -> List[torch.Tensor]:
        """Encode deltas between boxes and targets."""
        raise NotImplementedError

    @abc.abstractmethod
    def decode(
        self, boxes: List[Boxes2D], box_deltas: List[torch.Tensor]
    ) -> List[Boxes2D]:
        """Decode the predicted box_deltas according to given base boxes."""
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(  # type: ignore
        self,
        boxes: List[Boxes2D],
        targets: List[Boxes2D],
        box_deltas: List[torch.Tensor],
        intrinsics: Intrinsics,
    ) -> Union[List[torch.Tensor], List[Boxes2D]]:
        """Call."""
        raise NotImplementedError


class BaseBoxCoder3D(Vis4DModule[List[torch.Tensor], List[Boxes3D]]):
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

    @abc.abstractmethod
    def __call__(  # type: ignore
        self,
        boxes: List[Boxes2D],
        targets: List[Boxes3D],
        box_deltas: List[torch.Tensor],
        intrinsics: Intrinsics,
    ) -> Union[List[torch.Tensor], List[Boxes3D]]:
        """Call."""
        raise NotImplementedError
