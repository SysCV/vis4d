"""RPN tests."""
from tests.util import generate_features
from vis4d.op.detect.rpn import RPNHead


def test_rpn_head():
    """Test RPNHead class."""
    batch_size, num_anchors, wh, nc, num_feats = 2, 9, 128, 256, 5
    # default setup
    rpn_head = RPNHead(num_anchors, 2, nc, nc)
    test_features = [None, None] + generate_features(
        256, wh, wh, num_feats, batch_size
    )
    # rpn forward
    rpn_out = rpn_head(test_features)
    assert len(rpn_out.cls) == len(rpn_out.box) == num_feats
    for i in range(num_feats):
        wh_ = wh // 2**i
        assert rpn_out.cls[i].shape == (batch_size, num_anchors, wh_, wh_)
        assert rpn_out.box[i].shape == (batch_size, num_anchors * 4, wh_, wh_)
