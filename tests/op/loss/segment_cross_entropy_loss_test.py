"""Test SegmentCrossEntropyLoss."""
import torch

from vis4d.op.loss import SegmentCrossEntropyLoss


def test_segment_cross_entropy_loss():
    """Test SegmentCrossEntropyLoss."""
    loss_fn = SegmentCrossEntropyLoss()
    output = torch.rand(1, 3, 256, 256)
    target = torch.randint(0, 3, (1, 256, 256))
    loss = loss_fn(output, target)
    assert "loss_segment" in loss
    assert not torch.isnan(loss["loss_segment"])
    assert loss["loss_segment"] >= 0.0

    # test different size and dtype
    output = torch.rand(1, 3, 280, 280)
    target = torch.randint(0, 3, (1, 256, 256)).float()
    loss = loss_fn(output, target)
    assert "loss_segment" in loss
    assert not torch.isnan(loss["loss_segment"])
    assert loss["loss_segment"] >= 0.0
