"""Test MultiLevelSegLoss."""
import torch

from vis4d.op.loss import MultiLevelSegLoss


def test_multi_level_seg_loss():
    """Test MultiLevelSegLoss."""
    batch_size = 2
    num_classes = 3
    height = 10
    width = 10
    num_levels = 3
    outputs = [
        torch.rand(batch_size, num_classes, height, width)
        for _ in range(num_levels)
    ]
    target = torch.randint(0, num_classes, (batch_size, height, width))

    # first level
    loss = MultiLevelSegLoss()
    losses = loss(outputs, target)
    assert "loss_seg_level0" in losses
    assert not torch.isnan(losses["loss_seg_level0"])
    assert losses["loss_seg_level0"] >= 0.0

    # all level
    loss = MultiLevelSegLoss(feature_idx=(0, 1, 2))
    losses = loss(outputs, target)
    for i in range(3):
        name = f"loss_seg_level{i}"
        assert name in losses
        assert not torch.isnan(losses[name])
        assert losses[name] >= 0.0

    # all levels with weights
    weights = [1.0, 0.5, 0.25]
    loss = MultiLevelSegLoss(feature_idx=(0, 1, 2), weights=weights)
    losses_w = loss(outputs, target)
    for i in range(3):
        name = f"loss_seg_level{i}"
        assert name in losses_w
        assert not torch.isnan(losses_w[name])
        assert losses_w[name] >= 0.0
        assert losses_w[name] == weights[i] * losses[name]

    # two levels
    loss = MultiLevelSegLoss(feature_idx=(0, 2))
    losses = loss(outputs, target)
    assert "loss_seg_level0" in losses
    assert not torch.isnan(losses["loss_seg_level0"])
    assert losses["loss_seg_level0"] >= 0.0
    assert "loss_seg_level2" in losses
    assert not torch.isnan(losses["loss_seg_level2"])
    assert losses["loss_seg_level2"] >= 0.0
