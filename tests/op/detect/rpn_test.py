"""RPN tests."""
import torch

from tests.util import (
    generate_features,
    generate_features_determ,
    get_test_file,
)
from vis4d.common.ckpt import load_model_checkpoint
from vis4d.op.detect.rpn import RPNHead

REV_KEYS = [(r"^rpn_reg\.", "rpn_box.")]


def test_rpn_head():
    """Test RPNHead class."""
    # default setup
    batch_size, num_anchors, wh, nc, num_feats = 2, 3, 64, 256, 3
    rpn_head = RPNHead(num_anchors, 1, nc, nc)
    load_model_checkpoint(
        rpn_head, get_test_file("rpn_ckpt.pth"), rev_keys=REV_KEYS
    )
    test_features = [None, None] + generate_features_determ(
        256, wh, wh, num_feats, batch_size
    )
    # rpn forward
    rpn_out = rpn_head(test_features)
    rpn_gt = torch.load(get_test_file("rpn_gt.pth"))
    assert len(rpn_out.cls) == len(rpn_out.box) == num_feats
    for i in range(num_feats):
        wh_ = wh // 2**i
        assert rpn_out.cls[i].shape == (batch_size, num_anchors, wh_, wh_)
        assert rpn_out.box[i].shape == (batch_size, num_anchors * 4, wh_, wh_)
        assert torch.isclose(rpn_out.cls[i], rpn_gt.cls[i]).all()
        assert torch.isclose(rpn_out.box[i], rpn_gt.box[i]).all()

    # 2 convs
    batch_size, num_anchors, wh, nc, num_feats = 2, 3, 64, 32, 3
    rpn_head = RPNHead(num_anchors, 2, nc, nc)
    test_features = [None, None] + generate_features(
        32, wh, wh, num_feats, batch_size
    )
    # rpn forward
    rpn_out = rpn_head(test_features)
    assert len(rpn_out.cls) == len(rpn_out.box) == num_feats
    for i in range(num_feats):
        wh_ = wh // 2**i
        assert rpn_out.cls[i].shape == (batch_size, num_anchors, wh_, wh_)
        assert rpn_out.box[i].shape == (batch_size, num_anchors * 4, wh_, wh_)
