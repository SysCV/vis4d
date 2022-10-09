"""Resize augmentation."""
import random
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from numpy import block
from soupsieve import select

from vis4d.common import COMMON_KEYS, DictStrAny

# from ..datasets.base import COMMON_KEYS, DictData
from .base import Transform


@Transform(
    in_keys=[COMMON_KEYS.points3d],
    out_keys=[COMMON_KEYS.points3d],
)
def move_pts_to_last_channel():
    """TODO"""

    def _move_features_to_last_channel(*args: List[torch.Tensor]):
        if len(args) == 1:
            return args[0].transpose(-1, -2).contiguous()
        return [d.transpose(-1, -2).contiguous() for d in args]

    return _move_features_to_last_channel


@Transform(
    in_keys=[COMMON_KEYS.points3d, COMMON_KEYS.colors3d],
    out_keys=[COMMON_KEYS.points3d],
)
def concatenate_point_features():
    """TODO"""

    def _concatenate_point_features(*args: List[torch.Tensor]):
        return torch.cat(args)

    return _concatenate_point_features


@Transform(
    in_keys=[COMMON_KEYS.points3d, COMMON_KEYS.colors3d],
    out_keys=[COMMON_KEYS.points3d],
)
def center_and_normalize():
    """TODO"""

    def _center_and_normalize(coordinates: torch.Tensor):
        hwl = (
            torch.max(coordinates, dim=0).values
            - torch.min(coordinates, dim=0).values
        )
        center = torch.mean(coordinates, dim=0)
        return (coordinates - center) / (hwl / 2)

    return _center_and_normalize
