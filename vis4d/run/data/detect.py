"""Detect data module."""
from typing import List, Tuple, Union

from torch.utils.data import DataLoader, Dataset

from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms.base import compose, random_apply
from vis4d.data.transforms.flip import flip_boxes2d, flip_image
from vis4d.data.transforms.normalize import normalize_image
from vis4d.data.transforms.pad import pad_image
from vis4d.data.transforms.resize import (
    resize_boxes2d,
    resize_image,
    resize_masks,
)


def default_train_pipeline(
    datasets: Union[Dataset, List[Dataset]],
    batch_size: int,
    num_workers: int,
    im_hw: Tuple[int, int],
    with_mask: bool = False,
) -> DataLoader:
    """Default train preprocessing pipeline for detectors."""
    resize_trans = [resize_image(im_hw, keep_ratio=True), resize_boxes2d()]
    flip_trans = [flip_image(), flip_boxes2d()]
    if with_mask:
        resize_trans += [resize_masks()]
        flip_trans += [flip_masks()]
    preprocess_fn = compose(
        [*resize_trans, random_apply(flip_trans), normalize_image()]
    )
    batchprocess_fn = pad_image()
    datapipe = DataPipe(datasets, preprocess_fn)
    train_loader = build_train_dataloader(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=num_workers,
        batchprocess_fn=batchprocess_fn,
    )
    return train_loader


def default_test_pipeline(
    datasets: Union[Dataset, List[Dataset]],
    batch_size: int,
    num_workers: int,
    im_hw: Tuple[int, int],
) -> DataLoader:
    """Default test preprocessing pipeline for detectors."""
    preprocess_fn = compose(
        [
            resize_image(im_hw, keep_ratio=True, align_long_edge=True),
            normalize_image(),
        ]
    )
    batchprocess_fn = pad_image()
    datapipe = DataPipe(datasets, preprocess_fn)
    test_loaders = build_inference_dataloaders(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=num_workers,
        batchprocess_fn=batchprocess_fn,
    )
    return test_loaders
