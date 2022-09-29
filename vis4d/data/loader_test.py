"""Test loader components."""
import torch

from vis4d.data.datasets.base import DataKeys

from .datasets.coco import COCO
from .loader import (
    DataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
)
from .transforms import Normalize, Pad, Resize
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
