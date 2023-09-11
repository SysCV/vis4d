"""RCNN tests."""
import torch

from tests.util import generate_boxes, generate_features
from vis4d.op.box.encoder import DeltaXYWHBBoxDecoder
from vis4d.op.detect.rcnn import RCNNHead, RoI2Det


def test_rcnn_head():
    """Test RCNNHead class."""
    batch_size, num_classes, num_boxes, wh, inc = 2, 5, 10, 256, 64
    # default setup
    rcnn_head = RCNNHead(num_classes=num_classes, in_channels=64)
    test_features = [None, None] + generate_features(
        inc, wh, wh, 4, batch_size
    )
    boxes, _, _, _ = generate_boxes(wh * 4, wh * 4, num_boxes, batch_size)
    cls_score, bbox_pred = rcnn_head(test_features, boxes)
    assert len(cls_score) == len(bbox_pred) == num_boxes * batch_size
    assert cls_score.shape == (num_boxes * batch_size, num_classes + 1)
    assert bbox_pred.shape == (num_boxes * batch_size, num_classes * 4)

    # larger RoI size
    rcnn_head = RCNNHead(
        num_classes=num_classes,
        roi_size=(14, 14),
        in_channels=64,
        fc_out_channels=256,
    )
    boxes, _, _, _ = generate_boxes(1024, 1024, num_boxes, batch_size)
    cls_score, bbox_pred = rcnn_head(test_features, boxes)
    assert len(cls_score) == len(bbox_pred) == num_boxes * 2
    assert cls_score.shape == (num_boxes * 2, num_classes + 1)
    assert bbox_pred.shape == (num_boxes * 2, num_classes * 4)


def test_roi2det():
    """Test RoI2Det class."""
    # default setup
    batch_size, num_classes, num_boxes = 2, 5, 10
    max_h, max_w = 256, 512
    roi2det = RoI2Det(DeltaXYWHBBoxDecoder())
    boxes, _, _, _ = generate_boxes(max_h, max_w, num_boxes, batch_size)
    boxes, scores, class_ids = roi2det(
        torch.rand(batch_size * num_boxes, num_classes),
        torch.rand(batch_size * num_boxes, num_classes * 4),
        boxes,
        [(max_h, max_w)] * batch_size,
    )
    assert len(boxes) == len(scores) == len(class_ids) == batch_size
    for j in range(batch_size):
        assert boxes[j].shape[0] == scores[j].shape[0] == class_ids[j].shape[0]
        assert boxes[j].shape[1] == 4
        box_min = torch.logical_and(boxes[j][:, 0] >= 0, boxes[j][:, 1] >= 0)
        box_max = torch.logical_and(
            boxes[j][:, 2] <= max_w, boxes[j][:, 3] <= max_h
        )
        assert torch.logical_and(box_min, box_max).all()
