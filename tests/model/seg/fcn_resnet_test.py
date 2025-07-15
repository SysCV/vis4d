"""FCN ResNet tests."""

from __future__ import annotations

import unittest

import torch
from torch.optim.sgd import SGD

from tests.util import get_test_data, get_test_file
from vis4d.common.ckpt import load_model_checkpoint
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets import COCO
from vis4d.model.seg.fcn_resnet import REV_KEYS, FCNResNet
from vis4d.op.loss import MultiLevelSegLoss

from .common import get_test_dataloader, get_train_dataloader


class FCNResNetTest(unittest.TestCase):
    """FCN ResNet test class."""

    def test_inference(self) -> None:
        """Test inference of FCNResNet."""
        model = FCNResNet(base_model="resnet50", resize=(64, 64))
        dataset = COCO(
            get_test_data("coco_test"), split="train", use_pascal_voc_cats=True
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
        loss_fn = MultiLevelSegLoss(feature_idx=(4, 5), weights=[0.5, 1])
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        dataset = COCO(
            get_test_data("coco_test"), split="train", use_pascal_voc_cats=True
        )
        train_loader = get_train_dataloader(dataset, 2)
        model.train()

        # test two training steps
        batch = next(iter(train_loader))
        optimizer.zero_grad()
        out = model(batch[K.images])
        assert out.pred.shape == (2, 21, 64, 64)
        loss = loss_fn(out.outputs, batch[K.seg_masks])
        assert "loss_seg_level4" in loss and "loss_seg_level5" in loss
        total_loss = sum(loss.values())
        assert not torch.isnan(total_loss)
        total_loss.backward()
        optimizer.step()

        out = model(batch[K.images])
        assert out.pred.shape == (2, 21, 64, 64)
        loss = loss_fn(out.outputs, batch[K.seg_masks])
        assert "loss_seg_level4" in loss and "loss_seg_level5" in loss
        assert not torch.isnan(sum(loss.values()))
