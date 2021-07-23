"""Utilities for unit tests."""
import inspect
import os
from typing import List

import torch
from detectron2.data import DatasetCatalog, MetadataCatalog

from vist.struct import Boxes2D


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


def d2_data_reset(datasets: List[str]) -> None:
    """Delete all given dataset instances."""
    for ds in datasets:
        if ds in DatasetCatalog.keys():
            DatasetCatalog.remove(ds)
            MetadataCatalog.remove(ds)
