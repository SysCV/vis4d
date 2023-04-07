"""FCN ResNet tests."""
from __future__ import annotations

import unittest

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

from tests.util import get_test_file
from vis4d.common.ckpt import load_model_checkpoint
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets import COCO
from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms import mask, normalize, resize
from vis4d.data.transforms.base import compose
from vis4d.model.segment.fcn_resnet import REV_KEYS, FCNResNet, FCNResNetLoss


def get_train_dataloader(datasets: Dataset, batch_size: int) -> DataLoader:
    """Get data loader for training."""
    preprocess_fn = compose(
        [
            resize.GenerateResizeParameters((64, 64)),
            resize.ResizeImage(),
            resize.ResizeInstanceMasks(),
            normalize.NormalizeImage(),
            mask.ConvertInstanceMaskToSegmentationMask(),
        ]
    )
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_train_dataloader(
        datapipe, samples_per_gpu=batch_size, workers_per_gpu=1
    )


def get_test_dataloader(datasets: Dataset, batch_size: int) -> DataLoader:
    """Get data loader for testing."""
    preprocess_fn = compose(
        [
            resize.GenerateResizeParameters((64, 64)),
            resize.ResizeImage(),
            normalize.NormalizeImage(),
        ]
    )
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_inference_dataloaders(
        datapipe, samples_per_gpu=batch_size, workers_per_gpu=1
    )[0]


class FCNResNetTest(unittest.TestCase):
    """FCN ResNet test class."""

    def test_inference(self) -> None:
        """Test inference of FCNResNet."""
        model = FCNResNet(base_model="resnet50", resize=(64, 64))
        dataset = COCO(
            get_test_file("coco_test"),
            split="train",
            use_pascal_voc_cats=True,
            minimum_box_area=10,
        )
        test_loader = get_test_dataloader(dataset, 2)
        batch = next(iter(test_loader))
        weights = (
            "https://download.pytorch.org/models/"
            "fcn_resnet50_coco-1167a1af.pth"
        )
        load_model_checkpoint(model, weights, rev_keys=REV_KEYS)

        model.eval()
        with torch.no_grad():
            outs = model(batch[K.images])

        pred = outs.pred.argmax(1)
        testcase_gt = torch.load(get_test_file("fcn_resnet.pt"))
        assert torch.isclose(pred, testcase_gt, atol=1e-4).all().item()

    def test_train(self) -> None:
        """Test FCNResNet training."""
        model = FCNResNet(base_model="resnet50", resize=(64, 64))
        loss_fn = FCNResNetLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        dataset = COCO(
            get_test_file("coco_test"),
            split="train",
            use_pascal_voc_cats=True,
            minimum_box_area=10,
        )
        train_loader = get_train_dataloader(dataset, 2)
        model.train()

        # test two training steps
        batch = next(iter(train_loader))
        optimizer.zero_grad()
        out = model(batch[K.images])
        assert out.pred.shape == (2, 21, 64, 64)
        loss = loss_fn(out, batch[K.segmentation_masks])
        assert "level_4" in loss and "level_5" in loss
        total_loss = sum(loss.values())
        assert not torch.isnan(total_loss)
        total_loss.backward()
        optimizer.step()

        out = model(batch[K.images])
        assert out.pred.shape == (2, 21, 64, 64)
        loss = loss_fn(out, batch[K.segmentation_masks])
        assert "level_4" in loss and "level_5" in loss
        assert not torch.isnan(sum(loss.values()))
