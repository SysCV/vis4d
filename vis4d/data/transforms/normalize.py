"""Normalize Transform."""
from typing import Tuple

import torch

from vis4d.data.datasets.base import DataKeys, DictData
from vis4d.struct_to_revise import DictStrAny

from .base import Transform


def _normalize(
    img: torch.Tensor,
    pixel_mean: Tuple[float, float, float],
    pixel_std: Tuple[float, float, float],
) -> torch.Tensor:
    """Normalize tensor with given mean and std."""
    pixel_mean = torch.tensor(pixel_mean, device=img.device).view(-1, 1, 1)
    pixel_std = torch.tensor(pixel_std, device=img.device).view(-1, 1, 1)
    img = (img - pixel_mean) / pixel_std
    return img


class Normalize(Transform):
    """Image normalization transform."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
        std: Tuple[float, float, float] = (58.395, 57.12, 57.375),
        in_keys: Tuple[str, ...] = (DataKeys.images,),
        out_key: str = DataKeys.images,
    ):
        """Init."""
        super().__init__(in_keys, out_key)
        self.mean = mean
        self.std = std

    def call(self, image: Tensor) -> Tensor:
        """Normalize images."""
        return normalize(image, self.mean, self.std)


class Transform(object):
    """Decorator for preprocessing ops, which adds `inkey` and `outkey` arguments.
    Note: Only supports single-input single-output ops.
    """

    def __init__(
        self, in_default_kesy=["image"], outdefault="image", with_data=False
    ):
        self.indefault = indefault
        self.outdefault = outdefault
        self.with_data = with_data
        self.fn = None
        self.args = []

    def __call__(self, orig_get_pp_fn):
        self.fn = orig_get_pp_fn
        args = []
        for key in self.in_keys:
            key_parts = key.split("/")
            arg = data
            for p in key_parts:
                arg = arg[p]
            args.append(arg)
        data[self.out_key] = self.call(*args)

        def get_ikok_pp_fn(
            *args, key=None, inkey=self.indefault, outkey=self.outdefault, **kw
        ):

            orig_pp_fn = orig_get_pp_fn(*args, **kw)

            def _ikok_pp_fn(data):
                # Optionally allow the function to get the full data dict as aux input.
                if self.with_data:
                    data[key or outkey] = orig_pp_fn(*args, data=data)
                else:
                    data[key or outkey] = orig_pp_fn(*args)
                return data

            return _ikok_pp_fn

        return get_ikok_pp_fn

    def __repr__(self) -> str:
        pass


@Transform()
def normalize(
    mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
    std: Tuple[float, float, float] = (58.395, 57.12, 57.375),
):
    def _normalize(img: torch.Tensor) -> torch.Tensor:
        """Normalize tensor with given mean and std."""
        pixel_mean = torch.tensor(mean, device=img.device).view(-1, 1, 1)
        pixel_std = torch.tensor(std, device=img.device).view(-1, 1, 1)
        img = (img - pixel_mean) / pixel_std
        return img
