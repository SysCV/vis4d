"""YOLOX tests."""
from tests.util import generate_features_determ
from vis4d.op.detect.yolox import YOLOXHead


def test_yolox_head():
    """Test YOLOXHead class."""
    batch_size, num_classes, nc, wh = 2, 5, 3, 128
    # default setup
    yolox_head = YOLOXHead(num_classes, 64).eval()
    test_features = generate_features_determ(64, wh, wh, nc, batch_size)
    # test forward
    cls_score, bbox_pred, objectness = yolox_head(test_features)
    assert len(cls_score) == len(bbox_pred) == len(objectness)
    for i, (cls_score_i, bbox_pred_i, objectness_i) in enumerate(
        zip(cls_score, bbox_pred, objectness)
    ):
        wh_ = wh / 2**i
        assert cls_score_i.shape == (batch_size, num_classes, wh_, wh_)
        assert bbox_pred_i.shape == (batch_size, 4, wh_, wh_)
        assert objectness_i.shape == (batch_size, 1, wh_, wh_)
