"""Test loader components."""
import torch

from vis4d.data.datasets.base import DataKeys

from .datasets.coco import COCO, coco_seg_cats
from .datasets.s3dis import S3DIS
from .loader import (
    DataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
)
from .transforms import (
    ConvertInsMasksToSegMask,
    FilterByCategory,
    Normalize,
    Pad,
    RemapCategory,
    Resize,
    RandomPointSampler
)
from .transforms.base import batch_transform_pipeline, transform_pipeline


def test_train_loader():
    """Test the data loading pipeline."""
    coco = COCO(data_root="data/COCO/")
    batch_size = 2
    preprocess_fn = transform_pipeline(
        [Resize((256, 256), keep_ratio=True), Normalize()]
    )
    batchprocess_fn = batch_transform_pipeline([Pad()])

    datapipe = DataPipe(coco, preprocess_fn)
    train_loader = build_train_dataloader(
        datapipe, samples_per_gpu=batch_size, batchprocess_fn=batchprocess_fn
    )

    for sample in train_loader:
        assert isinstance(sample[DataKeys.images], torch.Tensor)
        assert batch_size == sample[DataKeys.images].size(0)
        assert batch_size == len(sample[DataKeys.boxes2d])
        break


def test_segment_train_loader():
    """Test the data loading pipeline."""
    coco = COCO(data_root="data/COCO/", with_mask=True)
    batch_size = 2
    preprocess_fn = transform_pipeline(
        [
            Resize((520, 520), keep_ratio=True),
            Normalize(),
            FilterByCategory(keep=coco_seg_cats),
            RemapCategory(mapping=coco_seg_cats),
            ConvertInsMasksToSegMask(),
        ]
    )
    batchprocess_fn = batch_transform_pipeline([Pad()])

    datapipe = DataPipe(coco, preprocess_fn)
    train_loader = build_train_dataloader(
        datapipe, samples_per_gpu=batch_size, batchprocess_fn=batchprocess_fn
    )

    for sample in train_loader:
        assert isinstance(sample[DataKeys.images], torch.Tensor)
        assert batch_size == sample[DataKeys.images].size(0)
        assert batch_size == len(sample[DataKeys.boxes2d])
        assert (
            sample[DataKeys.segmentation_mask].shape[-2:]
            == sample[DataKeys.images].shape[-2:]
        )
        break


def test_inference_loader():
    """Test the data loading pipeline."""
    coco = COCO(data_root="data/COCO/", split="val2017")
    preprocess_fn = transform_pipeline(
        [Resize((256, 256), keep_ratio=True), Normalize()]
    )
    batchprocess_fn = batch_transform_pipeline([Pad()])

    datapipe = DataPipe(coco, preprocess_fn)
    test_loaders = build_inference_dataloaders(
        datapipe, batchprocess_fn=batchprocess_fn
    )

    for sample in test_loaders[0]:
        assert isinstance(sample[DataKeys.images], torch.Tensor)
        assert 1 == sample[DataKeys.images].size(0)
        assert 1 == len(sample[DataKeys.boxes2d])
        break


def test_train_loader_3D():
    """Test the data loading pipeline for 3D Data."""
    s3dis = S3DIS(data_root="/data/Stanford3dDataset_v1.2")

    batch_size = 2
    preprocess_fn = transform_pipeline([RandomPointSampler(n_pts=1000)])

    batchprocess_fn = batch_transform_pipeline([])

    datapipe = DataPipe(s3dis, preprocess_fn)
    train_loader = build_train_dataloader(
        datapipe, samples_per_gpu=batch_size, batchprocess_fn=batchprocess_fn
    )

    for sample in train_loader:
        assert isinstance(sample[DataKeys.colors3d], torch.Tensor)
        assert isinstance(sample[DataKeys.points3d], torch.Tensor)
        assert isinstance(sample[DataKeys.semantics3d], torch.Tensor)
        assert batch_size == sample[DataKeys.colors3d].size(0)
        assert batch_size == sample[DataKeys.points3d].size(0)
        assert batch_size == sample[DataKeys.semantics3d].size(0)
        break
