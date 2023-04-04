"""Segment loss tests."""
from __future__ import annotations

import unittest

import torch
from torch import nn

from vis4d.op.segment.loss import SegmentLoss


class SegmentLossTest(unittest.TestCase):
    """SegmentLoss test class."""

    def test_loss(self):
        """Test loss function."""
        feature_idx = [4, 5]
        loss_fn = nn.CrossEntropyLoss()
        weights = [0.5, 1]
        loss = SegmentLoss(feature_idx, loss_fn, weights)

        outputs = [torch.randn(1, 10, 64, 64)] * 6
        target = torch.randint(0, 10, (1, 64, 64))
        losses = loss(outputs, target)
        assert len(losses) == 2
        assert "level_4" in losses
        assert "level_5" in losses

        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loss = SegmentLoss(feature_idx, loss_fn, weights)
        losses_w = loss(outputs, target)
        assert len(losses_w) == 2
        for l in ("level_4", "level_5"):
            assert l in losses_w
            assert losses_w[l].shape == (1, 64, 64)
            assert torch.isclose(losses_w[l].mean(), losses[l]).all()

        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loss = SegmentLoss(feature_idx, loss_fn)
        losses_w = loss(outputs, target)
        assert len(losses_w) == 2
        for i, l in enumerate(("level_4", "level_5")):
            assert l in losses_w
            assert losses_w[l].shape == (1, 64, 64)
            lmean = losses_w[l].mean() * 0.5 if i == 0 else losses_w[l].mean()
            assert torch.isclose(lmean, losses[l]).all()

        loss_fn = nn.CrossEntropyLoss()
        loss = SegmentLoss(feature_idx, loss_fn, [0.0, 0.0])
        losses_w = loss(outputs, target)
        assert len(losses_w) == 2
        for i, l in enumerate(("level_4", "level_5")):
            assert l in losses_w
            assert (losses_w[l] == 0).all()
