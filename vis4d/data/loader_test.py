"""Test loader components."""
import torch

from vis4d.data.const import COMMON_KEYS
from vis4d.data.datasets.coco import COCO
from vis4d.data.datasets.s3dis import S3DIS
from vis4d.data.loader import (
    DataPipe,
    SubdividingIterableDataset,
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms import compose, normalize, pad, resize


def test_train_loader():
    """Test the data loading pipeline."""
    coco = COCO(data_root="data/COCO/")
    batch_size = 2
    preprocess_fn = compose(
        [
            resize.resize_image((256, 256), keep_ratio=True),
            normalize.normalize_image(),
        ]
    )
    batchprocess_fn = compose([pad.pad_image()])

    datapipe = DataPipe(coco, preprocess_fn)
    train_loader = build_train_dataloader(
        datapipe, samples_per_gpu=batch_size, batchprocess_fn=batchprocess_fn
    )

    for sample in train_loader:
        assert isinstance(sample[COMMON_KEYS.images], torch.Tensor)
        assert batch_size == sample[COMMON_KEYS.images].size(0)
        assert batch_size == len(sample[COMMON_KEYS.boxes2d])
        break


def test_inference_loader():
    """Test the data loading pipeline."""
    coco = COCO(data_root="data/COCO/", split="val2017")
    preprocess_fn = compose(
        [
            resize.resize_image((256, 256), keep_ratio=True),
            normalize.normalize_image(),
        ]
    )
    batchprocess_fn = compose([pad.pad_image()])

    datapipe = DataPipe(coco, preprocess_fn)
    test_loaders = build_inference_dataloaders(
        datapipe, batchprocess_fn=batchprocess_fn
    )

    for sample in test_loaders[0]:
        assert isinstance(sample[COMMON_KEYS.images], torch.Tensor)
        assert 1 == sample[COMMON_KEYS.images].size(0)
        assert 1 == len(sample[COMMON_KEYS.boxes2d])
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
        assert isinstance(sample[COMMON_KEYS.images], torch.Tensor)
        assert batch_size == sample[COMMON_KEYS.images].size(0)
        assert batch_size == len(sample[COMMON_KEYS.boxes2d])
        assert (
            sample[COMMON_KEYS.segmentation_mask].shape[-2:]
            == sample[COMMON_KEYS.images].shape[-2:]
        )
        break


def test_train_loader_3D():
    """Test the data loading pipeline for 3D Data."""
    s3dis = S3DIS(data_root="/data/Stanford3dDataset_v1.2")

    batch_size = 2
    preprocess_fn = transform_pipeline([RandomPointSampler(n_pts=1024)])

    datapipe = DataPipe(s3dis, preprocess_fn)
    train_loader = build_train_dataloader(datapipe, samples_per_gpu=batch_size)

    for sample in train_loader:
        assert isinstance(sample[COMMON_KEYS.colors3d], torch.Tensor)
        assert isinstance(sample[COMMON_KEYS.points3d], torch.Tensor)
        assert isinstance(sample[COMMON_KEYS.semantics3d], torch.Tensor)
        assert isinstance(sample[COMMON_KEYS.instances3d], torch.Tensor)
        assert batch_size == sample[COMMON_KEYS.colors3d].size(0)
        assert batch_size == sample[COMMON_KEYS.points3d].size(0)
        assert batch_size == sample[COMMON_KEYS.semantics3d].size(0)
        break


def test_train_loader_3D_full_scene_batched():
    """Test the data loading pipeline for 3D Data with full scene sampling."""
    s3dis = S3DIS(data_root="/data/Stanford3dDataset_v1.2")

    batch_size = 2
    preprocess_fn = transform_pipeline(
        [FullCoverageBlockSampler(n_pts_per_block=1024)]
    )

    datapipe = DataPipe(s3dis, preprocess_fn)
    inference_loader = build_inference_dataloaders(
        SubdividingIterableDataset(datapipe, n_samples_per_batch=1024),
        samples_per_gpu=batch_size,
    )

    for sample in inference_loader[0]:
        assert isinstance(sample[COMMON_KEYS.colors3d], torch.Tensor)
        assert isinstance(sample[COMMON_KEYS.points3d], torch.Tensor)
        assert isinstance(sample[COMMON_KEYS.semantics3d], torch.Tensor)
        assert isinstance(sample[COMMON_KEYS.instances3d], torch.Tensor)
        assert isinstance(sample[COMMON_KEYS.index], torch.Tensor)

        assert batch_size == sample[COMMON_KEYS.colors3d].size(0)
        assert batch_size == sample[COMMON_KEYS.points3d].size(0)
        assert batch_size == sample[COMMON_KEYS.semantics3d].size(0)
        assert batch_size == sample[COMMON_KEYS.instances3d].size(0)
        assert batch_size == sample[COMMON_KEYS.index].size(0)
        break
