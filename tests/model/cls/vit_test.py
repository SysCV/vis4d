"""Faster RCNN tests."""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.imagenet import ImageNet
from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms.autoaugment import RandAug
from vis4d.data.transforms.base import compose, compose_batch
from vis4d.data.transforms.crop import (
    CropImage,
    GenCentralCropParameters,
    GenRandomSizeCropParameters,
)
from vis4d.data.transforms.mixup import (
    GenMixupParameters,
    MixupCategories,
    MixupImages,
)
from vis4d.data.transforms.normalize import NormalizeImage
from vis4d.data.transforms.random_erasing import RandomErasing
from vis4d.data.transforms.resize import GenerateResizeParameters, ResizeImage
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.model.cls import ClsOut
from vis4d.model.cls.vit import ViTClassifer


def get_train_dataloader(
    datasets: Dataset,
    batch_size: int,
    im_hw: tuple[int, int],
) -> DataLoader:
    """Get data loader for training."""
    random_resized_crop_trans = [
        GenRandomSizeCropParameters(),
        CropImage(),
        GenerateResizeParameters(im_hw, keep_ratio=False),
        ResizeImage(),
        RandAug(magnitude=10, use_increasing=True),
        RandomErasing(),
    ]
    mixup_trans = [
        GenMixupParameters(alpha=0.2, out_shape=im_hw),
        MixupImages(),
        MixupCategories(num_classes=2, label_smoothing=0.1),
    ]
    preprocess_fn = compose([*random_resized_crop_trans, NormalizeImage()])
    batchprocess_fn = compose_batch([*mixup_trans, ToTensor()])
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_train_dataloader(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=1,
        batchprocess_fn=batchprocess_fn,
    )


def get_test_dataloader(
    datasets: Dataset,
    batch_size: int,
) -> DataLoader:
    """Get data loader for testing."""
    preprocess_fn = compose(
        [
            GenerateResizeParameters(
                (256, 256),
                keep_ratio=True,
                allow_overflow=True,
            ),
            ResizeImage(),
            GenCentralCropParameters((224, 224)),
            CropImage(),
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

    def test_pretrained_vit_inference(self) -> None:
        """Test inference of ViTClassifer with timm's pretrained weights."""
        dataset = ImageNet(
            data_root=get_test_data("imagenet_1k_test"),
            split="train",
            keys_to_load=(
                K.images,
                K.categories,
            ),
            num_classes=2,
        )
        test_loader = get_test_dataloader(dataset, 3)

        vit_classifer = ViTClassifer(
            variant="vit_small_patch16_224",
            num_classes=1000,
            weights="timm://vit_small_patch16_224.augreg_in21k_ft_in1k",
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
        assert outs.probs.shape == (3, 1000)
        assert outs.logits.shape == (3, 1000)
        # 3 (the tiger shark) and 2 (great white shark) are the correct classes
        # for the test images.
        assert outs.probs.argmax(1).tolist() == [3, 3, 2]
        assert outs.logits.argmax(1).tolist() == [3, 3, 2]
        assert all(outs.logits[:, 3] > 7.0)

    def test_vit_training(self) -> None:
        """Test training of ViTClassifer."""
        dataset = ImageNet(
            data_root=get_test_data("imagenet_1k_test"),
            split="train",
            keys_to_load=(
                K.images,
                K.categories,
            ),
            num_classes=2,
        )
        train_loader = get_train_dataloader(dataset, 2, (224, 224))
        vit_classifer = ViTClassifer(
            variant="vit_small_patch16_224",
            num_classes=2,
            embed_dim=192,
            weights=None,
        )
        params = vit_classifer.parameters()
        vit_classifer.train()
        optimizer = torch.optim.SGD(params, lr=1e-4)
        batch = next(iter(train_loader))
        inputs, _ = (
            batch[K.images],
            batch[K.input_hw],
        )

        for _ in range(2):
            optimizer.zero_grad()
            outs = vit_classifer(inputs)
            loss = F.cross_entropy(outs.logits, batch[K.categories])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.01)
            optimizer.step()

        assert loss < 2.0

    def test_vit_training_2(self) -> None:
        """Test training of ViTClassifer."""
        dataset = ImageNet(
            data_root=get_test_data("imagenet_1k_test"),
            split="train",
            keys_to_load=(
                K.images,
                K.categories,
            ),
            num_classes=2,
        )
        train_loader = get_train_dataloader(dataset, 2, (224, 224))
        vit_classifer = ViTClassifer(
            variant="",
            num_classes=2,
            embed_dim=192,
            num_heads=3,
            class_token=False,
            use_global_pooling=True,
            weights=None,
        )
        params = vit_classifer.parameters()
        vit_classifer.train()
        optimizer = torch.optim.SGD(params, lr=1e-4)
        batch = next(iter(train_loader))
        inputs, _ = (
            batch[K.images],
            batch[K.input_hw],
        )

        for _ in range(2):
            optimizer.zero_grad()
            outs = vit_classifer(inputs)
            loss = F.cross_entropy(outs.logits, batch[K.categories])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.01)
            optimizer.step()

        assert loss < 3.0
