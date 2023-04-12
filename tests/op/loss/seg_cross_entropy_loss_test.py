"""Test SegCrossEntropyLoss."""
import torch

from vis4d.op.loss import SegCrossEntropyLoss


def test_seg_cross_entropy_loss():
    """Test SegCrossEntropyLoss."""
    loss_fn = SegCrossEntropyLoss()
    output = torch.rand(1, 3, 256, 256)
    target = torch.randint(0, 3, (1, 256, 256))
    loss = loss_fn(output, target)
    assert "loss_seg" in loss
    assert not torch.isnan(loss["loss_seg"])
    assert loss["loss_seg"] >= 0.0

    # test different size and dtype
    output = torch.rand(1, 3, 280, 280)
    target = torch.randint(0, 3, (1, 256, 256)).float()
    loss = loss_fn(output, target)
    assert "loss_seg" in loss
    assert not torch.isnan(loss["loss_seg"])
    assert loss["loss_seg"] >= 0.0
