"""Utilities for unit tests."""
from __future__ import annotations

import inspect
import os

import torch
from torch import nn


def get_test_file(file_name: str, rel_path: None | str = None) -> str:
    """Test test file path."""
    prefix = os.path.dirname(os.path.abspath(inspect.stack()[1][1]))
    prefix_code, prefix_rel = prefix.rsplit("vis4d", 1)
    if rel_path is None:
        rel_path = prefix_rel
    return os.path.join(
        prefix_code,
        "tests",
        rel_path.strip("/"),
        "testcases",
        file_name,
    )


def generate_features(
    channels: int,
    init_height: int,
    init_width: int,
    num_features: int,
    batch_size: int = 1,
    double_channels: bool = False,
) -> list[torch.Tensor]:
    """Create random feature lists."""
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(torch.manual_seed(0).get_state())

    features_list = []
    channel_factor = 1
    for i in range(num_features):
        features_list.append(
            torch.rand(
                batch_size,
                channels * channel_factor,
                init_height // (2**i),
                init_width // (2**i),
            )
        )
        if double_channels:
            channel_factor *= 2

    torch.random.set_rng_state(state)
    return features_list


def generate_boxes(
    height: int,
    width: int,
    num_boxes: int,
    batch_size: int = 1,
    track_ids: bool = False,
    use_score: bool = True,
):
    """Create random bounding boxes."""
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(torch.manual_seed(0).get_state())
    if use_score:
        box = [width, height, width, height, 1.0]
    else:
        box = [width, height, width, height]
    rand_max = torch.repeat_interleave(torch.tensor([box]), num_boxes, dim=0)
    box_tensor = torch.rand(num_boxes, 5 if use_score else 4) * rand_max
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
    tracks = torch.arange(0, num_boxes) if track_ids else None
    torch.random.set_rng_state(state)
    return (
        [box_tensor[:, :-1]] * batch_size,
        [box_tensor[:, -1:]] * batch_size,
        [torch.zeros(num_boxes, dtype=torch.long)] * batch_size,
        [tracks] * batch_size,
    )


def generate_masks(
    height: int,
    width: int,
    num_masks: int,
    batch_size: int = 1,
):
    """Create random masks."""
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(torch.manual_seed(0).get_state())
    rand_mask = torch.randint(0, num_masks, (height, width))
    mask_tensor = torch.stack(
        [(rand_mask == i).type(torch.uint8) for i in range(num_masks)]
    )
    torch.random.set_rng_state(state)
    return (
        [mask_tensor] * batch_size,
        [torch.rand(num_masks)] * batch_size,
        [torch.arange(num_masks)] * batch_size,
    )


# def generate_dets3d(
#     num_dets: int, track_ids: bool = False, use_score: bool = True
# ):
#     """Create random 3D detections."""
#     state = torch.random.get_rng_state()
#     torch.random.set_rng_state(torch.manual_seed(0).get_state())
#     min_inp = [-10, -1, 0, 1.5, 1, 3, 0, -math.pi, 0]
#     rand_min = torch.repeat_interleave(
#         torch.tensor([min_inp + [1.0] if use_score else min_inp]),
#         num_dets,
#         dim=0,
#     )
#     max_inp = [10, 1, 80, 2, 2, 4, 0, math.pi, 0]
#     rand_max = torch.repeat_interleave(
#         torch.tensor([max_inp + [1.0] if use_score else max_inp]),
#         num_dets,
#         dim=0,
#     )
#     box_tensor = (
#         torch.rand(num_dets, 10 if use_score else 9) * (rand_max - rand_min)
#         + rand_min
#     )

#     tracks = torch.arange(0, num_dets) if track_ids else None
#     dets = Boxes3D(box_tensor, torch.zeros(num_dets), tracks)
#     torch.random.set_rng_state(state)
#     return dets


class MockModel(nn.Module):
    """Model Mockup."""

    def __init__(self, model_param: int, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.model_param = model_param
        self.linear = nn.Linear(10, 1)

    def forward(self, *args, **kwargs):
        """Forward."""
        if self.training:
            return {
                "my_loss": (
                    self.linear(
                        torch.rand((1, 10), device=self.linear.weight.device)
                    )
                    - 0
                ).sum()
            }
        return {}  # type: ignore
