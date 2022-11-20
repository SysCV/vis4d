"""Faster RCNN tests."""
import unittest

import torch

from vis4d.op.box.encoder import DeltaXYWHBBoxEncoder
from vis4d.unittest.util import generate_boxes

from .rcnn import RCNNHead, RoI2Det


class RCNNTest(unittest.TestCase):
    """RCNN test class."""

    def test_rcnn_head(self):
        """Test RCNNHead class."""
        num_classes, num_boxes = 5, 10
        # default setup
        rcnn_head = RCNNHead(num_classes=num_classes, in_channels=64)
        test_features = [
            None,
            None,
            torch.rand(2, 64, 256, 256),
            torch.rand(2, 64, 128, 128),
            torch.rand(2, 64, 64, 64),
            torch.rand(2, 64, 32, 32),
        ]
        boxes, _, _, _ = generate_boxes(1024, 1024, num_boxes)
        cls_score, bbox_pred = rcnn_head(test_features, [boxes] * 2)
        assert len(cls_score) == len(bbox_pred) == num_boxes * 2
        assert cls_score.shape == (num_boxes * 2, num_classes + 1)
        assert bbox_pred.shape == (num_boxes * 2, num_classes * 4)

        # larger RoI size
        rcnn_head = RCNNHead(
            num_classes=num_classes,
            roi_size=(14, 14),
            in_channels=64,
            fc_out_channels=256,
        )
        boxes, _, _, _ = generate_boxes(1024, 1024, num_boxes)
        cls_score, bbox_pred = rcnn_head(test_features, [boxes] * 2)
        assert len(cls_score) == len(bbox_pred) == num_boxes * 2
        assert cls_score.shape == (num_boxes * 2, num_classes + 1)
        assert bbox_pred.shape == (num_boxes * 2, num_classes * 4)

    def test_roi2det(self):
        """Test RoI2Det class."""
        # default setup
        batch_size, num_classes, num_boxes = 2, 5, 10
        max_h, max_w = 256, 512
        roi2det = RoI2Det(DeltaXYWHBBoxEncoder())
        boxes, _, _, _ = generate_boxes(max_h, max_w, num_boxes)
        boxes, scores, class_ids = roi2det(
            torch.rand(batch_size * num_boxes, num_classes),
            torch.rand(batch_size * num_boxes, num_classes * 4),
            [boxes] * batch_size,
            [(max_h, max_w)] * batch_size,
        )
        assert len(boxes) == len(scores) == len(class_ids) == batch_size
        for j in range(batch_size):
            assert (
                boxes[0].shape[0]
                == scores[0].shape[0]
                == class_ids[0].shape[0]
            )
            assert boxes[j].shape[1] == 4
            box_min = torch.logical_and(
                boxes[j][:, 0] >= 0, boxes[j][:, 1] >= 0
            )
            box_max = torch.logical_and(
                boxes[j][:, 2] <= max_w, boxes[j][:, 3] <= max_h
            )
            assert torch.logical_and(box_min, box_max).all()
