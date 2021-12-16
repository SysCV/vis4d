"""RoI Pooling module base."""
import abc
from typing import List, Tuple

import torch

from vis4d.common.registry import RegistryHolder
from vis4d.struct import Boxes2D


class BaseRoIPooler(metaclass=RegistryHolder):
    """Base class for RoI poolers."""

    def __init__(self, resolution: Tuple[int, int]):
        """Init."""
        super().__init__()
        self.resolution = resolution

    @abc.abstractmethod
    def pool(
        self, features: List[torch.Tensor], boxes: List[Boxes2D]
    ) -> torch.Tensor:
        """Pool features in input bounding boxes from given feature maps."""
        raise NotImplementedError
