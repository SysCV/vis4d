"""Utilities for unit tests."""
import inspect
import math
import os
from typing import List

import torch

from vist.struct import Bitmasks, Boxes2D, Boxes3D


def get_test_file(file_name: str) -> str:
    """Test test file path."""
    return os.path.join(
        os.path.dirname(os.path.abspath(inspect.stack()[1][1])),
        "testcases",
        file_name,
    )


def generate_dets(
    height: int, width: int, num_dets: int, track_ids: bool = False
) -> Boxes2D:
    """Create random detections."""
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(torch.manual_seed(0).get_state())
    rand_max = torch.repeat_interleave(
        torch.tensor([[width, height, width, height, 1.0]]), num_dets, dim=0
    )
    box_tensor = torch.rand(num_dets, 5) * rand_max
    sorted_xy = [
        box_tensor[:, [0, 2]].sort(dim=-1)[0],
        box_tensor[:, [1, 3]].sort(dim=-1)[0],
    ]
    box_tensor[:, :4] = torch.cat(
        [
            sorted_xy[0][:, 0:1],
            sorted_xy[1][:, 0:1],
            sorted_xy[0][:, 1:2],
            sorted_xy[1][:, 1:2],
        ],
        dim=-1,
    )
    tracks = torch.arange(0, num_dets) if track_ids else None
    dets = Boxes2D(box_tensor, torch.zeros(num_dets), tracks)
    torch.random.set_rng_state(state)
    return dets


def generate_dets3d(num_dets: int, track_ids: bool = False) -> Boxes3D:
    """Create random 3D detections."""
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(torch.manual_seed(0).get_state())
    rand_min = torch.repeat_interleave(
        torch.tensor([[-10, -1, 0, 1.5, 1, 3, 0, -math.pi, 0, 1.0]]),
        num_dets,
        dim=0,
    )
    rand_max = torch.repeat_interleave(
        torch.tensor([[10, 1, 80, 2, 2, 4, 0, math.pi, 0, 1.0]]),
        num_dets,
        dim=0,
    )
    box_tensor = torch.rand(num_dets, 10) * (rand_max - rand_min) + rand_min

    tracks = torch.arange(0, num_dets) if track_ids else None
    dets = Boxes3D(box_tensor, torch.zeros(num_dets), tracks)
    torch.random.set_rng_state(state)
    return dets


def generate_masks(
    height: int, width: int, num_masks: int, track_ids: bool = False
) -> Bitmasks:
    """Create random masks."""
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(torch.manual_seed(0).get_state())
    mask_tensor = (torch.rand(num_masks, width, height) > 0.5).type(
        torch.uint8
    )
    tracks = torch.arange(0, num_masks) if track_ids else None
    masks = Bitmasks(
        mask_tensor, torch.zeros(num_masks), tracks, torch.rand(num_masks)
    )
    torch.random.set_rng_state(state)
    return masks


def generate_feature_list(
    channels: int, init_height: int, init_width: int, list_len: int
) -> List[torch.Tensor]:
    """Create random feature lists."""
    features_list = []
    torch.random.set_rng_state(torch.manual_seed(0).get_state())

    for i in range(list_len):
        features_list.append(
            torch.rand(
                1, channels, init_height // (2 ** i), init_width // (2 ** i)
            )
        )

    return features_list
