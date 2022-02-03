"""Utilities for augmentation."""
from typing import Tuple
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from vis4d.struct import Images
from enum import Enum


def sample_bernoulli(num: int, prob: float) -> torch.Tensor:
    """Sample from a Bernoulli distribution with given p."""
    curr_prob: torch.Tensor
    if prob == 1.0:
        curr_prob = torch.tensor([True] * num)
    elif prob == 0.0:
        curr_prob = torch.tensor([False] * num)
    else:
        curr_prob = Bernoulli(prob).sample((num,)).bool()
    return curr_prob


def sample_batched(num: int, prob: float, same: bool = False) -> torch.Tensor:
    """Sample num / 1 times from a Bernoulli distribution with given p."""
    if same:
        return sample_bernoulli(1, prob).repeat(num)
    return sample_bernoulli(num, prob)


def get_resize_shape(ori_wh: Tuple[int, int], new_wh: Tuple[int, int], keep_ratio: bool = True) -> Tuple[int, int]:
    """Get shape for resize, considering keep_ratio and long edge."""
    w, h = ori_wh
    new_w, new_h = new_wh
    if keep_ratio:
        long_edge, short_edge = max(new_wh), min(new_wh)
        scale_factor = min(
            long_edge / max(h, w), short_edge / min(h, w)
        )
        new_h = int(h * scale_factor + 0.5)
        new_w = int(w * scale_factor + 0.5)
    return new_w, new_h
    # elif w < h and not new_w < new_h or w > h and not new_w > new_h:
    #     # if long edge in original image is different from long edge in
    #     # resize shape, we flip it to avoid large image distortions
    #     return new_h, new_w


def im_resize(images: Images, resize_hw: Tuple[int, int], mode: str = "bilinear") -> Images:
    """Resize an Images object."""
    align_corners = None if mode == "nearest" else False
    resized_ims = []
    for im in images:
        im_t = F.interpolate(
            im.tensor,
            resize_hw,
            mode=mode,
            align_corners=align_corners,
        )
        resized_ims.append(Images(im_t, [(im_t.shape[3], im_t.shape[2])]))
    return Images.cat(resized_ims)


class FlipMode(Enum):
    """Enum defining the axis for horizontal / vertical flip."""
    HORIZONTAL = 3
    VERTICAL = 2


def im_flip(images: Images, mode: FlipMode = FlipMode.HORIZONTAL) -> Images:
    """Flip an Images object."""
    resized_ims = []
    for im in images:
        im_f = im.tensor.flip(mode.value)
        resized_ims.append(Images(im_f, im.image_sizes))
    return Images.cat(resized_ims)
