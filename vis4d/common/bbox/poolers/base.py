"""RoI Pooling module base."""
import abc
from typing import List, Tuple

import torch

from vis4d.common.module import Vis4DModule
from vis4d.struct import Boxes2D


class BaseRoIPooler(Vis4DModule[torch.Tensor, torch.Tensor]):
    """Base class for RoI poolers."""

    def __init__(self, resolution: Tuple[int, int]) -> None:
        """Init."""
        super().__init__()
        self.resolution = resolution

    @abc.abstractmethod
    def __call__(  # type: ignore
        self, features: List[torch.Tensor], boxes: List[Boxes2D]
    ) -> torch.Tensor:
        """Pool features in input bounding boxes from given feature maps."""
        raise NotImplementedError
