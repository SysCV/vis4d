"""ViT classification tests."""
from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from tests.util import get_test_data
from vis4d.config.example.vit_imagenet import get_config
from vis4d.data.const import CommonKeys
from vis4d.data.datasets import ImageNet
from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.normalize import normalize_image
from vis4d.data.transforms.resize import resize_image
from vis4d.engine.cli import _train as cli_train
from vis4d.model.classification.common import ClsOut
from vis4d.model.classification.vit import ClassificationViT


def get_train_dataloader(
    datasets: Dataset,
    batch_size: int,
    im_hw: tuple[int, int],
) -> DataLoader:
    """Get data loader for training."""
    resize_trans = [resize_image(im_hw, keep_ratio=False)]
    preprocess_fn = compose([*resize_trans, normalize_image()])
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_train_dataloader(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=1,
    )


def get_test_dataloader(
    datasets: Dataset, batch_size: int, im_hw: tuple[int, int]
) -> DataLoader:
    """Get data loader for testing."""
    preprocess_fn = compose([resize_image(im_hw), normalize_image()])
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_inference_dataloaders(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=1,
    )[0]


class ViTTest(unittest.TestCase):
    """ViT test class."""

    def test_inference(self) -> None:
        """Test inference of ViT Classification."""
        dataset = ImageNet(
            data_root=get_test_data("imagenet_1k_test"),
            keys_to_load=(CommonKeys.images, CommonKeys.categories),
            split="train",
            num_classes=2,
            use_sample_lists=False,
        )
        test_loader = get_test_dataloader(dataset, 2, (224, 224))
        model = ClassificationViT(num_classes=2, vit_name="vit_b_16")
        model.eval()

        batch = next(iter(test_loader))
        images = batch[CommonKeys.images]
        with torch.no_grad():
            out = model(images)

        assert isinstance(out, ClsOut)
        self.assertEqual(out.logits.shape, (2, 2))
        self.assertEqual(out.probs.shape, (2, 2))

    def test_cli_training(self) -> None:
        """Test ViT training via CLI."""
        config = get_config()
        config.num_epochs = 2
        config.n_gpus = 0

        cli_train(config)
