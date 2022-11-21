"""Test loader components."""
import torch

from vis4d.data.const import CommonKeys
from vis4d.data.datasets.coco import COCO
from vis4d.data.datasets.s3dis import S3DIS
from vis4d.data.loader import (
    DataPipe,
    SubdividingIterableDataset,
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms import compose, mask, normalize, pad, resize


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
        assert isinstance(sample[CommonKeys.images], torch.Tensor)
        assert batch_size == sample[CommonKeys.images].size(0)
        assert batch_size == len(sample[CommonKeys.boxes2d])
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
        assert isinstance(sample[CommonKeys.images], torch.Tensor)
        assert 1 == sample[CommonKeys.images].size(0)
        assert 1 == len(sample[CommonKeys.boxes2d])
        break


def test_segment_train_loader():
    """Test the data loading pipeline."""
    coco = COCO(
        data_root="data/COCO/", use_pascal_voc_cats=True, minimum_box_area=10
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
        images = sample[CommonKeys.images]
        segmentation_masks = sample[CommonKeys.segmentation_masks]

        assert isinstance(images, torch.Tensor)
        assert isinstance(segmentation_masks, torch.Tensor)
        assert 1 == images.size(0)
        assert 1 == segmentation_masks.size(0)
        assert segmentation_masks.shape[-2:] == images.shape[-2:]
        assert segmentation_masks.min() >= 0
        assert segmentation_masks[segmentation_masks != 255].max() <= 20
        break


def test_segment_inference_loader():
    """Test the data loading pipeline."""
    coco = COCO(
        data_root="data/COCO/", use_pascal_voc_cats=True, minimum_box_area=10
    )
    batch_size = 1
    preprocess_fn = compose(
        [
            normalize.normalize_image(),
            mask.convert_to_seg_masks(),
        ]
    )
    datapipe = DataPipe(coco, preprocess_fn)
    test_loader = build_inference_dataloaders(datapipe)

    for sample in test_loader[0]:
        images = sample[CommonKeys.images]
        segmentation_masks = sample[CommonKeys.segmentation_masks]

        assert isinstance(images, torch.Tensor)
        assert isinstance(segmentation_masks, torch.Tensor)
        assert batch_size == images.size(0)
        assert batch_size == segmentation_masks.size(0)
        assert segmentation_masks.shape[-2:] == images.shape[-2:]
        assert segmentation_masks.min() >= 0
        assert segmentation_masks[segmentation_masks != 255].max() <= 20
        break


def test_train_loader_3D():
    """Test the data loading pipeline for 3D Data."""
    s3dis = S3DIS(data_root="/data/Stanford3dDataset_v1.2")

    batch_size = 2
    preprocess_fn = transform_pipeline([RandomPointSampler(n_pts=1024)])

    datapipe = DataPipe(s3dis, preprocess_fn)
    train_loader = build_train_dataloader(datapipe, samples_per_gpu=batch_size)

    for sample in train_loader:
        assert isinstance(sample[CommonKeys.colors3d], torch.Tensor)
        assert isinstance(sample[CommonKeys.points3d], torch.Tensor)
        assert isinstance(sample[CommonKeys.semantics3d], torch.Tensor)
        assert isinstance(sample[CommonKeys.instances3d], torch.Tensor)
        assert batch_size == sample[CommonKeys.colors3d].size(0)
        assert batch_size == sample[CommonKeys.points3d].size(0)
        assert batch_size == sample[CommonKeys.semantics3d].size(0)
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
        assert isinstance(sample[CommonKeys.colors3d], torch.Tensor)
        assert isinstance(sample[CommonKeys.points3d], torch.Tensor)
        assert isinstance(sample[CommonKeys.semantics3d], torch.Tensor)
        assert isinstance(sample[CommonKeys.instances3d], torch.Tensor)
        assert isinstance(sample[CommonKeys.index], torch.Tensor)

        assert batch_size == sample[CommonKeys.colors3d].size(0)
        assert batch_size == sample[CommonKeys.points3d].size(0)
        assert batch_size == sample[CommonKeys.semantics3d].size(0)
        assert batch_size == sample[CommonKeys.instances3d].size(0)
        assert batch_size == sample[CommonKeys.index].size(0)
        break
