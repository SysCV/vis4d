"""Test loader components."""
from __future__ import annotations

import os
import unittest

import numpy as np
import torch

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.bdd100k import BDD100K
from vis4d.data.datasets.coco import COCO
from vis4d.data.datasets.s3dis import S3DIS
from vis4d.data.loader import (
    DataPipe,
    SubdividingIterableDataset,
    VideoDataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
    default_collate,
)
from vis4d.data.reference import UniformViewSampler
from vis4d.data.transforms import (
    compose,
    compose_batch,
    mask,
    normalize,
    pad,
    resize,
    to_tensor,
)
from vis4d.data.transforms.point_sampling import (
    GenerateBlockSamplingIndices,
    GenFullCovBlockSamplingIndices,
    SampleColors,
    SampleInstances,
    SamplePoints,
    SampleSemantics,
)
from vis4d.data.typing import DictData


class DataLoaderTest(unittest.TestCase):
    """Test loader components."""

    def test_datapipe(self) -> None:
        """Test the data loading pipeline."""
        dataset = BDD100K(
            data_root=os.path.join(
                get_test_data("bdd100k_test"), "track/images"
            ),
            annotation_path=os.path.join(
                get_test_data("bdd100k_test"), "track/labels"
            ),
            config_path="box_track",
        )
        datapipe = VideoDataPipe(
            dataset, reference_view_sampler=UniformViewSampler(1, 1)
        )
        batch = datapipe[0]
        assert len(batch) == 2
        assert set(batch[0].keys()) == {
            "images",
            "input_hw",
            "original_images",
            "original_hw",
            "axis_mode",
            "frame_ids",
            "sample_names",
            "sequence_names",
            "boxes2d",
            "boxes2d_classes",
            "boxes2d_track_ids",
        }
        assert batch[0]["frame_ids"] - batch[1]["frame_ids"] in [1, -1]

        batch = datapipe.get_dataset_sample_index(-1)
        with self.assertRaises(ValueError):
            datapipe.get_dataset_sample_index(-13)

    def test_train_loader(self) -> None:
        """Test the data loading pipeline."""
        coco = COCO(data_root=get_test_data("coco_test"), split="train")
        batch_size = 2
        preprocess_fn = compose(
            [
                resize.GenerateResizeParameters((256, 256), keep_ratio=True),
                resize.ResizeImage(),
                normalize.NormalizeImage(),
            ]
        )
        batchprocess_fn = compose_batch(
            [pad.PadImages(), to_tensor.ToTensor()]
        )

        datapipe = DataPipe(coco, preprocess_fn)
        train_loader = build_train_dataloader(
            datapipe,
            samples_per_gpu=batch_size,
            batchprocess_fn=batchprocess_fn,
        )

        for sample in train_loader:
            assert isinstance(sample[K.images], torch.Tensor)
            assert batch_size == sample[K.images].size(0)
            assert batch_size == len(sample[K.boxes2d])
            assert sample[K.boxes2d][0].shape[1] == 4
            assert sample[K.images].shape[1] == 3
            break

    def test_inference_loader(self) -> None:
        """Test the data loading pipeline."""
        coco = COCO(data_root=get_test_data("coco_test"), split="train")
        preprocess_fn = compose(
            [
                resize.GenerateResizeParameters((256, 256), keep_ratio=True),
                resize.ResizeImage(),
                normalize.NormalizeImage(),
            ]
        )
        batchprocess_fn = compose_batch(
            [pad.PadImages(), to_tensor.ToTensor()]
        )

        datapipe = DataPipe(coco, preprocess_fn)
        test_loaders = build_inference_dataloaders(
            datapipe, batchprocess_fn=batchprocess_fn
        )

        for sample in test_loaders[0]:
            assert isinstance(sample[K.images], torch.Tensor)
            assert sample[K.images].size(0) == 1
            assert len(sample[K.boxes2d]) == 1
            assert sample[K.boxes2d][0].shape == torch.Size([14, 4])
            assert sample[K.images].shape == torch.Size([1, 3, 192, 256])
            break

    def test_default_collate(self) -> None:
        """Test the default collate."""
        t1, t2 = torch.rand(2, 3), torch.rand(2, 3)
        data = [{K.seg_masks: t1}, {K.seg_masks: t2}]
        col_data = default_collate(data)
        assert (col_data[K.seg_masks] == torch.stack([t1, t2])).all()

        col_data = default_collate([{"name": ["a"]}, {"name": ["b"]}])
        assert col_data["name"] == [["a"], ["b"]]

        with self.assertRaises(RuntimeError):
            default_collate(
                [
                    {K.seg_masks: torch.rand(1, 3)},
                    {K.seg_masks: torch.rand(2, 3)},
                ]
            )


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
            resize.GenerateResizeParameters((520, 520)),
            resize.ResizeImage(),
            resize.ResizeInstanceMasks(),
            normalize.NormalizeImage(),
            mask.ConvertInstanceMaskToSegMask(),
        ]
    )
    batchprocess_fn = compose_batch([to_tensor.ToTensor()])
    datapipe = DataPipe(coco, preprocess_fn)
    train_loader = build_train_dataloader(
        datapipe, samples_per_gpu=batch_size, batchprocess_fn=batchprocess_fn
    )

    for sample in train_loader:
        images = sample[K.images]
        seg_masks = sample[K.seg_masks]

        assert isinstance(images, torch.Tensor)
        assert isinstance(seg_masks, torch.Tensor)
        assert images.size(0) == 2
        assert seg_masks.size(0) == 2
        assert seg_masks.shape[-2:] == images.shape[-2:]
        assert seg_masks.min() >= 0
        assert seg_masks[seg_masks != 255].max() <= 20
        assert images.shape == torch.Size([2, 3, 520, 520])
        assert seg_masks.shape == torch.Size([2, 520, 520])
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
        [
            normalize.NormalizeImage(),
            mask.ConvertInstanceMaskToSegMask(),
        ]
    )
    batchprocess_fn = compose_batch([to_tensor.ToTensor()])
    datapipe = DataPipe(coco, preprocess_fn)
    test_loader = build_inference_dataloaders(
        datapipe, batchprocess_fn=batchprocess_fn
    )

    for sample in test_loader[0]:
        images = sample[K.images]
        seg_masks = sample[K.seg_masks]

        assert isinstance(images, torch.Tensor)
        assert isinstance(seg_masks, torch.Tensor)
        assert batch_size == images.size(0)
        assert batch_size == seg_masks.size(0)
        assert seg_masks.shape[-2:] == images.shape[-2:]
        assert seg_masks.min() >= 0
        assert seg_masks[seg_masks != 255].max() <= 20
        assert images.shape == torch.Size([1, 3, 230, 352])
        assert seg_masks.shape == torch.Size([1, 230, 352])
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
            K.points3d,
            K.colors3d,
            K.instances3d,
            K.semantics3d,
        ):
            data[key] = np.stack([b[key] for b in batch], 0)
        else:
            data[key] = [b[key] for b in batch]
    return data


def test_train_loader_3d() -> None:
    """Test the data loading pipeline for 3D Data."""
    s3dis = S3DIS(data_root=get_test_data("s3d_test"))
    batch_size = 2
    preprocess_fn = compose(
        [
            GenerateBlockSamplingIndices(
                num_pts=1024, block_dimensions=(1, 1, 4)
            ),
            SampleInstances(),
            SampleSemantics(),
            SampleColors(),
            SamplePoints(),
        ]
    )

    datapipe = DataPipe(s3dis, preprocess_fn)
    train_loader = build_train_dataloader(
        datapipe, samples_per_gpu=batch_size, collate_fn=point_collate
    )

    for sample in train_loader:
        assert isinstance(sample[K.colors3d], np.ndarray)
        assert isinstance(sample[K.points3d], np.ndarray)
        assert isinstance(sample[K.semantics3d], np.ndarray)
        assert isinstance(sample[K.instances3d], np.ndarray)
        assert batch_size == sample[K.colors3d].shape[0]
        assert batch_size == sample[K.points3d].shape[0]
        assert batch_size == sample[K.semantics3d].shape[0]

        assert sample[K.semantics3d].shape[1] == 1024
        assert sample[K.points3d].shape[1] == 1024
        assert sample[K.colors3d].shape[1] == 1024
        assert sample[K.instances3d].shape[1] == 1024
        break


def test_train_loader_3d_batched() -> None:
    """Test the data loading pipeline for 3D Data with full scene sampling."""
    s3dis = S3DIS(data_root=get_test_data("s3d_test"))
    keys = (K.points3d, K.colors3d, K.instances3d, K.semantics3d)
    batch_size = 2
    preprocess_fn = compose(
        [
            GenFullCovBlockSamplingIndices(
                num_pts=1024, block_dimensions=(1, 1, 4)
            ),
            SampleInstances(),
            SampleSemantics(),
            SampleColors(),
            SamplePoints(),
        ]
    )

    datapipe = DataPipe(s3dis, preprocess_fn)
    inference_loader = build_inference_dataloaders(
        SubdividingIterableDataset(datapipe, n_samples_per_batch=1024),
        samples_per_gpu=batch_size,
        collate_fn=point_collate,
    )

    for sample in inference_loader[0]:
        for k in keys:
            assert isinstance(sample[k], np.ndarray)
            assert batch_size == sample[k].shape[0]
            assert sample[k].shape[1] == 1024
        break
