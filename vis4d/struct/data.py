"""Vis4D Input data structures."""

import itertools
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .structures import InputInstance

# TODO restructure, move to data utilies or common


def inverse(self) -> torch.Tensor:
    """Invert rigid transformation matrix [R^T, -R^T * t]."""
    rot = self.rotation.permute(0, 2, 1)
    t = -rot @ self.translation
    inv = torch.cat([torch.cat([rot, t], -1), self.tensor[:, 3:4]], 1)
    return inv


def get_image(self, inputdata: InputData, idx: int) -> "Images":
    """Access the individual image in its original size.

    Args:
        idx: int or slice

    Returns:
        Tensor: an image of shape (C_1, ..., C_K, H, W)
        where K >= 1
    """
    size = self.image_sizes[idx]
    return Images(
        self.tensor[idx : idx + 1, ..., : size[1], : size[0]], [size]
    )


def batch_images(
    cls,
    instances: List[torch.Tensor],
    device: Optional[torch.device] = None,
    stride: int = 32,
) -> torch.Tensor:
    """Concatenate and pad."""
    assert isinstance(instances, (list, tuple))
    assert len(instances) > 0
    assert all((isinstance(inst, Images) for inst in instances))
    max_hw = (
        max([im.tensor.shape[-2] for im in instances]),
        max([im.tensor.shape[-1] for im in instances]),
    )
    lens = [len(x) for x in instances]

    # ensure divisibility by stride
    pad = lambda x: (x + (cls.stride - 1)) // cls.stride * cls.stride
    max_hw = (pad(x) for x in max_hw)  # type: ignore

    batch_shape = (
        [sum(lens)] + list(instances[0].tensor.shape[1:-2]) + list(max_hw)
    )
    if device is None:
        device = instances[0].tensor.device
    pad_imgs = torch.zeros(batch_shape, device=device)
    cum_len = 0
    for img, cur_len in zip(instances, lens):
        pad_imgs[
            cum_len : cum_len + cur_len,
            ...,
            : img.tensor.shape[-2],
            : img.tensor.shape[-1],
        ].copy_(img.tensor)
        cum_len += cur_len

    all_sizes = list(
        itertools.chain.from_iterable([x.image_sizes for x in instances])
    )
    return Images(pad_imgs.contiguous(), all_sizes)


def resize(
    self, resize_hw: Tuple[int, int], mode: str = "bilinear"
) -> None:  # TODO adjust
    """Resizes Images object."""
    align_corners = None if mode == "nearest" else False
    resized_ims = []
    for i in range(len(self)):
        w, h = self.image_sizes[i]
        im_t = F.interpolate(
            self.tensor[i : i + 1, ..., :h, :w],
            resize_hw,
            mode=mode,
            align_corners=align_corners,
        )
        resized_ims.append(im_t)
        self.image_sizes[i] = (im_t.shape[3], im_t.shape[2])
    self.tensor = torch.cat(resized_ims)


class FlipMode(Enum):
    """Enum defining the axis for horizontal / vertical flip."""

    HORIZONTAL = 3
    VERTICAL = 2


def flip(self, mode: FlipMode = FlipMode.HORIZONTAL) -> None:  # TODO adjust
    """Flips Images object."""
    for i in range(len(self)):
        w, h = self.image_sizes[i]
        self.tensor[i : i + 1, ..., :h, :w] = self.tensor[
            i : i + 1, ..., :h, :w
        ].flip(mode.value)


def batch_pointclouds(
    cls,
    instances: List["PointCloud"],
    device: Optional[torch.device] = None,
) -> torch.Tensor:  # TODO adjust
    """Concatenate N PointCloud objects into Padded foramt.

    Returns:
        Tensor: [Batch, N_max, num_point_feature].
    """
    assert isinstance(instances, (list, tuple))
    assert len(instances) > 0

    if device is None:
        device = instances[0].tensor.device

    max_points = max([p.tensor.shape[1] for p in instances])
    max_feature = max([p.tensor.shape[2] for p in instances])

    tot_batch = sum([len(inst) for inst in instances])

    pad_points = torch.zeros(
        (tot_batch, max_points, max_feature), device=device
    )

    cum_len = 0
    for inst in instances:
        cur_len = inst.tensor.shape[0]
        pad_points[
            cum_len : cum_len + cur_len, : inst.tensor.shape[1], :
        ] = inst.tensor
        cum_len += cur_len
    return pad_points
