"""Faster RCNN tests."""
from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.imagenet import ImageNet
from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms.base import compose, compose_batch
from vis4d.data.transforms.normalize import NormalizeImage
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.resize import (
    GenResizeParameters,
    ResizeImage,
    ResizeInstanceMasks,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.model.cls import ClsOut
from vis4d.model.cls.vit import ViTClassifer


def get_train_dataloader(
    datasets: Dataset,
    batch_size: int,
    im_hw: tuple[int, int],
    with_mask: bool = False,
) -> DataLoader:
    """Get data loader for training."""
    resize_trans = [
        GenResizeParameters(im_hw, keep_ratio=True),
        ResizeImage(),
    ]
    if with_mask:
        resize_trans += [ResizeInstanceMasks()]
    preprocess_fn = compose([*resize_trans, NormalizeImage()])
    batchprocess_fn = compose_batch([PadImages(), ToTensor()])
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_train_dataloader(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=1,
        batchprocess_fn=batchprocess_fn,
    )


def get_test_dataloader(
    datasets: Dataset, batch_size: int, im_hw: tuple[int, int]
) -> DataLoader:
    """Get data loader for testing."""
    preprocess_fn = compose(
        [
            GenResizeParameters(im_hw),
            ResizeImage(),
            NormalizeImage(),
        ]
    )
    batchprocess_fn = compose_batch([ToTensor()])
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_inference_dataloaders(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=1,
        batchprocess_fn=batchprocess_fn,
    )[0]


class ViTClassifierTest(unittest.TestCase):
    """ViTClassifier test class."""

    def test_inference(self) -> None:
        """Test inference of ViTClassifer."""
        dataset = ImageNet(
            data_root=get_test_data("imagenet_1k_test"),
            split="train",
            keys_to_load=(
                K.images,
                K.categories,
            ),
            num_classes=2,
        )
        test_loader = get_test_dataloader(dataset, 2, (512, 512))

        vit_classifer = ViTClassifer(
            variant="vit_small_patch16_224",
            num_classes=1000,
            pretrained=True,
        )
        vit_classifer.eval()

        batch = next(iter(test_loader))
        inputs, _ = (
            batch[K.images],
            batch[K.input_hw],
        )

        with torch.no_grad():
            outs = vit_classifer(inputs)

        assert isinstance(outs, ClsOut)

    # TODO: add test for training after refactoring config
