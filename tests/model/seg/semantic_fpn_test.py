"""Semantic FPN tests."""
from __future__ import annotations

import unittest

import torch
from torch import optim

from tests.util import get_test_data, get_test_file
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets import COCO
from vis4d.model.seg.semantic_fpn import SemanticFPN
from vis4d.op.loss import SegCrossEntropyLoss

from .common import get_test_dataloader, get_train_dataloader


class SemanticFPNTest(unittest.TestCase):
    """Semantic FPN test class."""

    dataset = COCO(
        get_test_data("coco_test"),
        split="train",
        use_pascal_voc_cats=True,
        minimum_box_area=10,
    )

    def test_inference(self) -> None:
        """Test inference of SemanticFPN."""
        state = torch.random.get_rng_state()
        torch.random.set_rng_state(torch.manual_seed(0).get_state())
        model = SemanticFPN(num_classes=21, weights="bdd100k")
        test_loader = get_test_dataloader(self.dataset, 2)
        batch = next(iter(test_loader))

        model.eval()
        with torch.no_grad():
            outs = model(batch[K.images], batch["original_hw"])

        pred = outs.masks
        testcase_gt = torch.load(get_test_file("semantic_fpn.pt"))
        for p, g in zip(pred, testcase_gt):
            assert torch.isclose(p, g, atol=1e-4).all().item()

        torch.random.set_rng_state(state)

    def test_train(self) -> None:
        """Test SemanticFPN training."""
        model = SemanticFPN(num_classes=21)
        loss_fn = SegCrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        train_loader = get_train_dataloader(self.dataset, 2)
        model.train()

        # test two training steps
        batch = next(iter(train_loader))
        optimizer.zero_grad()
        out = model(batch[K.images])
        assert out.outputs.shape == (2, 21, 64, 64)
        loss = loss_fn(out.outputs, batch[K.seg_masks])
        assert "loss_seg" in loss
        assert not torch.isnan(loss["loss_seg"])
        total_loss = sum(loss.values())
        total_loss.backward()
        optimizer.step()

        out = model(batch[K.images])
        assert out.outputs.shape == (2, 21, 64, 64)
        loss = loss_fn(out.outputs, batch[K.seg_masks])
        assert "loss_seg" in loss
        assert not torch.isnan(loss["loss_seg"])
