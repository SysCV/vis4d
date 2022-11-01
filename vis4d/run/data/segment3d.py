"""Segment3D data module."""
from typing import List, Union

import torch
from torch.utils.data import DataLoader, Dataset

from vis4d.data.const import COMMON_KEYS
from vis4d.data.loader import (
    DataPipe,
    SubdividingIterableDataset,
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.point_sampling import (
    sample_points_block_full_coverage,
    sample_points_block_random,
)
from vis4d.data.transforms.points import (
    add_norm_noise,
    center_and_normalize,
    concatenate_point_features,
    extract_pc_bounds,
    move_pts_to_last_channel,
    normalize_by_bounds,
    rotate_around_axis,
)
from vis4d.data.typing import DictData


def default_collate(batch: List[DictData]) -> DictData:
    """Default batch collate."""
    data = {}
    for key in batch[0]:
        data[key] = torch.stack([b[key] for b in batch], 0)
    return data


def default_train_pipeline(
    datasets: Union[Dataset, List[Dataset]],
    batch_size: int,
    num_pts: int = 2048,
    load_instances: bool = False,
    load_colors: bool = False,
) -> DataLoader:
    """Default train preprocessing pipeline for 3D segmentation."""
    data_keys = [COMMON_KEYS.points3d]
    labels_keys = [COMMON_KEYS.semantics3d]

    if load_instances:
        labels_keys.append(COMMON_KEYS.instances3d)
    if load_colors:
        data_keys.append(COMMON_KEYS.colors3d)

    bounds_calc = extract_pc_bounds()
    sample = sample_points_block_random(
        in_keys=data_keys + labels_keys,
        out_keys=data_keys + labels_keys,
        num_pts=num_pts,
        min_pts=512,
    )

    noise = add_norm_noise(std=0.02)
    rand_rotate_z = rotate_around_axis(axis=2)
    norm = center_and_normalize(
        in_keys=[COMMON_KEYS.points3d],
        out_keys=["points3d_normalized"],
        normalize=False,
    )
    bounds_norm = normalize_by_bounds()

    data_keys += ["points3d_normalized"]
    move_pts = move_pts_to_last_channel(in_keys=data_keys, out_keys=data_keys)

    pipeline = [
        # noise,
        # rand_rotate_z,
        bounds_calc,
        sample,
        norm,
        bounds_norm,
        move_pts,
    ]

    if len(data_keys) > 1:
        pipeline.append(
            concatenate_point_features(
                in_keys=data_keys, out_keys=[COMMON_KEYS.points3d]
            )
        )
    preprocess_fn = compose(pipeline)

    datapipe = DataPipe(datasets, preprocess_fn)
    train_loader = build_train_dataloader(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=12,
        collate_fn=default_collate,
    )
    return train_loader


def default_test_pipeline(
    datasets: Union[Dataset, List[Dataset]],
    batch_size: int,
    num_pts: int = 2048,
    load_instances: bool = False,
    load_colors: bool = False,
) -> DataLoader:
    """Default test preprocessing pipeline for 3D segmentation."""
    data_keys = [COMMON_KEYS.points3d]
    labels_keys = [COMMON_KEYS.semantics3d]

    if load_instances:
        labels_keys.append(COMMON_KEYS.instances3d)
    if load_colors:
        data_keys.append(COMMON_KEYS.colors3d)

    sample = sample_points_block_full_coverage(
        in_keys=data_keys + labels_keys,
        out_keys=data_keys + labels_keys,
        n_pts_per_block=num_pts,
        min_pts_per_block=512,
    )
    bounds_calc = extract_pc_bounds()

    norm = center_and_normalize(
        in_keys=[COMMON_KEYS.points3d],
        out_keys=["points3d_normalized"],
        normalize=False,
    )
    bounds_norm = normalize_by_bounds()

    move_pts = move_pts_to_last_channel(in_keys=data_keys, out_keys=data_keys)
    pipeline = [norm, bounds_norm, move_pts]
    data_keys += ["points3d_normalized"]

    if len(data_keys) > 1:
        pipeline.append(
            concatenate_point_features(
                in_keys=data_keys, out_keys=[COMMON_KEYS.points3d]
            )
        )
    preprocess_fn = compose(pipeline)
    datapipe = SubdividingIterableDataset(
        DataPipe(datasets, compose([bounds_calc, sample])),
        n_samples_per_batch=num_pts,
        preprocess_fn=preprocess_fn,
    )

    test_loaders = build_inference_dataloaders(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=12,
        collate_fn=default_collate,
    )
    return test_loaders
