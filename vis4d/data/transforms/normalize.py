"""Normalize Transform."""
from typing import Tuple

import torch

from vis4d.data.datasets.base import DataKeys, DictData
from vis4d.struct_to_revise import DictStrAny

from .base import BaseTransform


def normalize(
    img: torch.Tensor,
    pixel_mean: Tuple[float, float, float],
    pixel_std: Tuple[float, float, float],
) -> torch.Tensor:
    """Normalize tensor with given mean and std."""
    pixel_mean = torch.tensor(pixel_mean, device=img.device).view(-1, 1, 1)
    pixel_std = torch.tensor(pixel_std, device=img.device).view(-1, 1, 1)
    img = (img - pixel_mean) / pixel_std
    return img


class Normalize(BaseTransform):
    """Image normalization transform."""

    def __init__(
        self,
        in_keys: Tuple[str, ...] = (DataKeys.images,),
        mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
        std: Tuple[float, float, float] = (58.395, 57.12, 57.375),
    ):
        """Init."""
        super().__init__(in_keys)
        self.mean = mean
        self.std = std

    def generate_parameters(self, data: DictData) -> DictStrAny:
        """Generate parameters (empty)."""
        return {}

    def __call__(self, data: DictData, parameters: DictStrAny) -> DictData:
        """Normalize images."""
        data[DataKeys.images] = normalize(
            data[DataKeys.images], self.mean, self.std
        )
        return data
