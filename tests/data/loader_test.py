# pylint: disable=unexpected-keyword-arg
# TODO remove this once new transforms are implemented
"""Test loader components."""
from __future__ import annotations

import torch

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as Keys
from vis4d.data.datasets.coco import COCO
from vis4d.data.datasets.s3dis import S3DIS
from vis4d.data.loader import (
    DataPipe,
    SubdividingIterableDataset,
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms import compose, mask, normalize, pad, resize
from vis4d.data.transforms.point_sampling import (
    sample_points_block_full_coverage,
    sample_points_random,
)
from vis4d.data.typing import DictData


def test_train_loader() -> None:
    """Test the data loading pipeline."""
    coco = COCO(data_root=get_test_data("coco_test"), split="train")
    batch_size = 2
    preprocess_fn = compose(
        [
            resize.resize_image((256, 256), keep_ratio=True),
            normalize.normalize_image(),
        ]
    )
    batchprocess_fn = pad.pad_image()

    datapipe = DataPipe(coco, preprocess_fn)
    train_loader = build_train_dataloader(
        datapipe, samples_per_gpu=batch_size, batchprocess_fn=batchprocess_fn
    )

    for sample in train_loader:
        assert isinstance(sample[Keys.images], torch.Tensor)
        assert batch_size == sample[Keys.images].size(0)
        assert batch_size == len(sample[Keys.boxes2d])
        break


def test_inference_loader() -> None:
    """Test the data loading pipeline."""
    coco = COCO(data_root=get_test_data("coco_test"), split="train")
    preprocess_fn = compose(
        [
            resize.resize_image((256, 256), keep_ratio=True),
            normalize.normalize_image(),
        ]
    )
    batchprocess_fn = pad.pad_image()

    datapipe = DataPipe(coco, preprocess_fn)
    test_loaders = build_inference_dataloaders(
        datapipe, batchprocess_fn=batchprocess_fn
    )

    for sample in test_loaders[0]:
        assert isinstance(sample[Keys.images], torch.Tensor)
        assert sample[Keys.images].size(0) == 1
        assert len(sample[Keys.boxes2d]) == 1
        break


def test_segment_train_loader() -> None:
    """Test the data loading pipeline."""
    coco = COCO(
        data_root=get_test_data("coco_test"),
        split="train",
        use_pascal_voc_cats=True,
        minimum_box_area=10,
    )
    batch_size = 4
    preprocess_fn = compose(
        [
            resize.resize_image((520, 520)),
            resize.resize_masks(),
            normalize.normalize_image(),
            mask.convert_to_seg_masks(),
        ]
    )
    datapipe = DataPipe(coco, preprocess_fn)
    train_loader = build_train_dataloader(datapipe, samples_per_gpu=batch_size)

    for sample in train_loader:
        images = sample[Keys.images]
        segmentation_masks = sample[Keys.segmentation_masks]

        assert isinstance(images, torch.Tensor)
        assert isinstance(segmentation_masks, torch.Tensor)
        assert images.size(0) == 2
        assert segmentation_masks.size(0) == 2
        assert segmentation_masks.shape[-2:] == images.shape[-2:]
        assert segmentation_masks.min() >= 0
        assert segmentation_masks[segmentation_masks != 255].max() <= 20
        break


def test_segment_inference_loader() -> None:
    """Test the data loading pipeline."""
    coco = COCO(
        data_root=get_test_data("coco_test"),
        split="train",
        use_pascal_voc_cats=True,
        minimum_box_area=10,
    )
    batch_size = 1
    preprocess_fn = compose(
        [normalize.normalize_image(), mask.convert_to_seg_masks()]
    )
    datapipe = DataPipe(coco, preprocess_fn)
    test_loader = build_inference_dataloaders(datapipe)

    for sample in test_loader[0]:
        images = sample[Keys.images]
        segmentation_masks = sample[Keys.segmentation_masks]

        assert isinstance(images, torch.Tensor)
        assert isinstance(segmentation_masks, torch.Tensor)
        assert batch_size == images.size(0)
        assert batch_size == segmentation_masks.size(0)
        assert segmentation_masks.shape[-2:] == images.shape[-2:]
        assert segmentation_masks.min() >= 0
        assert segmentation_masks[segmentation_masks != 255].max() <= 20
        break


def point_collate(batch: list[DictData]) -> DictData:
    """Stacks point samples at the first axis of the tensor.

    Args:
        batch (list[DictData]): List with data from the dataset.

    Returns:
        DictData: Collated data.
    """
    data = {}
    for key in batch[0]:
        if key in (
            Keys.points3d,
            Keys.colors3d,
            Keys.instances3d,
            Keys.semantics3d,
        ):
            data[key] = torch.stack([b[key] for b in batch], 0)
        else:
            data[key] = [b[key] for b in batch]
    return data


def test_train_loader_3d() -> None:
    """Test the data loading pipeline for 3D Data."""
    s3dis = S3DIS(data_root=get_test_data("s3d_test"))

    batch_size = 2
    keys = (
        Keys.points3d,
        Keys.colors3d,
        Keys.instances3d,
        Keys.semantics3d,
    )
    preprocess_fn = compose(
        [sample_points_random(in_keys=keys, out_keys=keys, num_pts=1024)]
    )

    datapipe = DataPipe(s3dis, preprocess_fn)
    train_loader = build_train_dataloader(
        datapipe, samples_per_gpu=batch_size, collate_fn=point_collate
    )

    for sample in train_loader:
        assert isinstance(sample[Keys.colors3d], torch.Tensor)
        assert isinstance(sample[Keys.points3d], torch.Tensor)
        assert isinstance(sample[Keys.semantics3d], torch.Tensor)
        assert isinstance(sample[Keys.instances3d], torch.Tensor)
        assert batch_size == sample[Keys.colors3d].size(0)
        assert batch_size == sample[Keys.points3d].size(0)
        assert batch_size == sample[Keys.semantics3d].size(0)
        break


def test_train_loader_3d_batched() -> None:
    """Test the data loading pipeline for 3D Data with full scene sampling."""
    s3dis = S3DIS(data_root=get_test_data("s3d_test"))
    keys = (
        Keys.points3d,
        Keys.colors3d,
        Keys.instances3d,
        Keys.semantics3d,
    )
    batch_size = 2
    preprocess_fn = compose(
        [
            sample_points_block_full_coverage(
                in_keys=keys, out_keys=keys, n_pts_per_block=1024
            )
        ]
    )

    datapipe = DataPipe(s3dis, preprocess_fn)
    inference_loader = build_inference_dataloaders(
        SubdividingIterableDataset(datapipe, n_samples_per_batch=1024),
        samples_per_gpu=batch_size,
        collate_fn=point_collate,
    )

    for sample in inference_loader[0]:
        assert isinstance(sample[Keys.colors3d], torch.Tensor)
        assert isinstance(sample[Keys.points3d], torch.Tensor)
        assert isinstance(sample[Keys.semantics3d], torch.Tensor)
        assert isinstance(sample[Keys.instances3d], torch.Tensor)

        assert batch_size == sample[Keys.colors3d].size(0)
        assert batch_size == sample[Keys.points3d].size(0)
        assert batch_size == sample[Keys.semantics3d].size(0)
        assert batch_size == sample[Keys.instances3d].size(0)
        break
