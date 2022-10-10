"""Detect data module."""
from typing import List, Tuple, Union

from torch.utils.data import DataLoader, Dataset

from vis4d.common.typing import COMMON_KEYS
from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.point_sampling import sample_points_block_random
from vis4d.data.transforms.points import (
    center_and_normalize,
    concatenate_point_features,
    move_pts_to_last_channel,
)


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

    sample = sample_points_block_random(
        in_keys=data_keys + labels_keys,
        out_keys=data_keys + labels_keys,
        num_pts=num_pts,
    )
    norm = center_and_normalize(
        in_keys=[COMMON_KEYS.points3d], out_keys=["points3d_normalized"]
    )
    data_keys += ["points3d_normalized"]
    move_pts = move_pts_to_last_channel(in_keys=data_keys, out_keys=data_keys)

    pipeline = [sample, norm, move_pts]

    if len(data_keys) > 1:
        pipeline.append(
            concatenate_point_features(
                in_keys=data_keys, out_keys=[COMMON_KEYS.points3d]
            )
        )
    preprocess_fn = compose(pipeline)

    datapipe = DataPipe(datasets, preprocess_fn)
    train_loader = build_train_dataloader(datapipe, samples_per_gpu=batch_size)
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

    sample = sample_points_block_random(
        in_keys=data_keys + labels_keys,
        out_keys=data_keys + labels_keys,
        num_pts=num_pts,
    )
    norm = center_and_normalize(
        in_keys=[COMMON_KEYS.points3d], out_keys=["points3d_normalized"]
    )
    move_pts = move_pts_to_last_channel(in_keys=data_keys, out_keys=data_keys)
    pipeline = [sample, norm, move_pts]
    data_keys += ["points3d_normalized"]

    if len(data_keys) > 1:
        pipeline.append(
            concatenate_point_features(
                in_keys=data_keys, out_keys=[COMMON_KEYS.points3d]
            )
        )
    preprocess_fn = compose(pipeline)

    datapipe = DataPipe(datasets, preprocess_fn)
    test_loaders = build_inference_dataloaders(
        datapipe, samples_per_gpu=batch_size
    )
    return test_loaders
