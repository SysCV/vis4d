"""Utilities for unit tests."""
import inspect
import math
import os
from typing import List

import torch
from scalabel.label.typing import Frame, ImageSize

from vis4d.struct import (
    Boxes2D,
    Boxes3D,
    Images,
    InputSample,
    InstanceMasks,
    Intrinsics,
    LabelInstances,
    SemanticMasks,
)


def get_test_file(file_name: str) -> str:
    """Test test file path."""
    return os.path.join(
        os.path.dirname(os.path.abspath(inspect.stack()[1][1])),
        "testcases",
        file_name,
    )


def generate_dets(
    height: int,
    width: int,
    num_dets: int,
    track_ids: bool = False,
    use_score: bool = True,
) -> Boxes2D:
    """Create random detections."""
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(torch.manual_seed(0).get_state())
    rand_max = torch.repeat_interleave(
        torch.tensor(
            [
                [width, height, width, height, 1.0]
                if use_score
                else [width, height, width, height]
            ]
        ),
        num_dets,
        dim=0,
    )
    box_tensor = torch.rand(num_dets, 5 if use_score else 4) * rand_max
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
    dets = Boxes2D(box_tensor, torch.zeros(num_dets, dtype=torch.long), tracks)
    torch.random.set_rng_state(state)
    return dets


def generate_dets3d(
    num_dets: int, track_ids: bool = False, use_score: bool = True
) -> Boxes3D:
    """Create random 3D detections."""
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(torch.manual_seed(0).get_state())
    min_inp = [-10, -1, 0, 1.5, 1, 3, 0, -math.pi, 0]
    rand_min = torch.repeat_interleave(
        torch.tensor([min_inp + [1.0] if use_score else min_inp]),
        num_dets,
        dim=0,
    )
    max_inp = [10, 1, 80, 2, 2, 4, 0, math.pi, 0]
    rand_max = torch.repeat_interleave(
        torch.tensor([max_inp + [1.0] if use_score else max_inp]),
        num_dets,
        dim=0,
    )
    box_tensor = (
        torch.rand(num_dets, 10 if use_score else 9) * (rand_max - rand_min)
        + rand_min
    )

    tracks = torch.arange(0, num_dets) if track_ids else None
    dets = Boxes3D(box_tensor, torch.zeros(num_dets), tracks)
    torch.random.set_rng_state(state)
    return dets


def generate_instance_masks(
    height: int, width: int, num_masks: int, track_ids: bool = False
) -> InstanceMasks:
    """Create random masks."""
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(torch.manual_seed(0).get_state())
    mask_tensor = (torch.rand(num_masks, width, height) > 0.5).type(
        torch.uint8
    )
    tracks = torch.arange(0, num_masks) if track_ids else None
    masks = InstanceMasks(
        mask_tensor, torch.zeros(num_masks), tracks, torch.rand(num_masks)
    )
    torch.random.set_rng_state(state)
    return masks


def generate_semantic_masks(
    height: int, width: int, num_masks: int
) -> SemanticMasks:
    """Create random masks."""
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(torch.manual_seed(0).get_state())
    rand_mask = torch.randint(0, num_masks, (width, height))
    mask_tensor = torch.stack(
        [(rand_mask == i).type(torch.uint8) for i in range(num_masks)]
    )
    masks = SemanticMasks(mask_tensor, torch.arange(num_masks))
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


def generate_input_sample(
    height: int,
    width: int,
    num_imgs: int,
    num_objs: int,
    track_ids: bool = False,
    det_input: bool = True,
    det3d_input: bool = False,
    pan_input: bool = False,
    use_score: bool = True,
) -> InputSample:
    """Create random InputSample."""
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(torch.manual_seed(0).get_state())
    image_tensor = (torch.rand(num_imgs, 3, height, width) * 255).type(
        torch.float32
    )
    images = Images(image_tensor, [(width, height)] * num_imgs)
    sample = InputSample(
        [
            Frame(
                name="test_frame",
                frameIndex=0,
                size=ImageSize(width=width, height=height),
            )
        ]
        * num_imgs,
        images,
    )
    sample.intrinsics = Intrinsics.cat(
        [Intrinsics(torch.eye(3)) for _ in range(num_imgs)]
    )
    targets = {}
    if det_input or pan_input:
        targets["boxes2d"] = [
            generate_dets(height, width, num_objs, track_ids, use_score)
        ] * num_imgs
        targets["instance_masks"] = [
            generate_instance_masks(height, width, num_objs, track_ids)
        ] * num_imgs
    if det3d_input:
        targets["boxes3d"] = [
            generate_dets3d(num_objs, track_ids, use_score)
        ] * num_imgs
    if not det_input or pan_input:
        targets["semantic_masks"] = [
            generate_semantic_masks(height, width, num_objs)
        ] * num_imgs
    sample.targets = LabelInstances(**targets)
    torch.random.set_rng_state(state)
    return sample
