"""Pad transformation."""
from turtle import width
from typing import List, Tuple

import torch.nn.functional as F

from vis4d.data.datasets.base import DataKeys, DictData
from vis4d.struct_to_revise import DictStrAny

from .base import BaseBatchTransform, DataKeys


class Pad(BaseBatchTransform):
    """Pad image to fit certain stride."""

    def __init__(
        self,
        in_keys: Tuple[str, ...] = (DataKeys.images,),
        stride: int = 32,
        mode: str = "constant",
        value: float = 0.0,
    ):
        super().__init__(in_keys)
        self.stride = stride
        self.mode = mode
        self.value = value

    def generate_parameters(self, data: List[DictData]) -> DictStrAny:
        """Generate padding parameters."""
        heights = [d[DataKeys.images].shape[-2] for d in data]
        widths = [d[DataKeys.images].shape[-1] for d in data]
        max_hw = max(heights), max(widths)

        # ensure divisibility by stride
        pad = lambda x: (x + (self.stride - 1)) // self.stride * self.stride
        max_hw = tuple(pad(x) for x in max_hw)

        # generate params for torch pad
        pad_params = []
        for h, w in zip(heights, widths):
            pad_param = (0, max_hw[1] - w, 0, max_hw[0] - h)
            pad_params.append(pad_param)
        return dict(pad_params=pad_params)

    def __call__(
        self, data: List[DictData], parameters: DictStrAny
    ) -> List[DictData]:
        """Apply padding to batch."""
        for d, pad_param in zip(data, parameters["pad_params"]):
            d[DataKeys.images] = F.pad(
                d[DataKeys.images], pad_param, self.mode, self.value
            )
        return data
