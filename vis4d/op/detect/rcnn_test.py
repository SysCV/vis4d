"""RCNN tests."""
import unittest

import torch

from vis4d.op.box.encoder import DeltaXYWHBBoxEncoder
from vis4d.unittest.util import generate_boxes

from .rcnn import Det2Mask, MaskRCNNHead, RCNNHead, RoI2Det


class RCNNTest(unittest.TestCase):
    """RCNN test class."""

    def test_rcnn_head(self):
        """Test RCNNHead class."""
        batch_size, num_classes, num_boxes, wh, inc = 2, 5, 10, 256, 64
        # default setup
        rcnn_head = RCNNHead(num_classes=num_classes, in_channels=64)
        test_features = [None, None] + [
            torch.rand(batch_size, inc, wh // 2**i, wh // 2**i)
            for i in range(4)
        ]
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

    def test_roi2det(self):
        """Test RoI2Det class."""
        # default setup
        batch_size, num_classes, num_boxes = 2, 5, 10
        max_h, max_w = 256, 512
        roi2det = RoI2Det(DeltaXYWHBBoxEncoder())
        boxes, _, _, _ = generate_boxes(max_h, max_w, num_boxes, batch_size)
        boxes, scores, class_ids = roi2det(
            torch.rand(batch_size * num_boxes, num_classes),
            torch.rand(batch_size * num_boxes, num_classes * 4),
            boxes,
            [(max_h, max_w)] * batch_size,
        )
        assert len(boxes) == len(scores) == len(class_ids) == batch_size
        for j in range(batch_size):
            assert (
                boxes[j].shape[0]
                == scores[j].shape[0]
                == class_ids[j].shape[0]
            )
            assert boxes[j].shape[1] == 4
            box_min = torch.logical_and(
                boxes[j][:, 0] >= 0, boxes[j][:, 1] >= 0
            )
            box_max = torch.logical_and(
                boxes[j][:, 2] <= max_w, boxes[j][:, 3] <= max_h
            )
            assert torch.logical_and(box_min, box_max).all()


class MaskRCNNTest(unittest.TestCase):
    """Mask RCNN test class."""

    def test_mask_rcnn_head(self):
        """Test MaskRCNNHead class."""
        batch_size, num_classes, num_boxes, wh, inc = 2, 5, 10, 256, 64
        # default setup
        rcnn_head = MaskRCNNHead(num_classes=num_classes, in_channels=64)
        test_features = [None, None] + [
            torch.rand(batch_size, inc, wh // 2**i, wh // 2**i)
            for i in range(4)
        ]
        boxes, _, _, _ = generate_boxes(wh * 4, wh * 4, num_boxes, batch_size)
        mask_pred = rcnn_head(test_features, boxes).mask_pred
        assert len(mask_pred) == batch_size
        for mask in mask_pred:
            assert mask.shape == (num_boxes, num_classes, 14 * 2, 14 * 2)

        # smaller RoI size
        rcnn_head = MaskRCNNHead(
            num_classes=num_classes,
            roi_size=(7, 7),
            in_channels=64,
            conv_out_channels=128,
        )
        boxes, _, _, _ = generate_boxes(wh * 4, wh * 4, num_boxes, batch_size)
        mask_pred = rcnn_head(test_features, boxes).mask_pred
        assert len(mask_pred) == batch_size
        for mask in mask_pred:
            assert mask.shape == (num_boxes, num_classes, 7 * 2, 7 * 2)

    def test_det2mask(self):
        """Test Det2Mask class."""
        # default setup
        batch_size, num_classes, num_boxes = 2, 5, 10
        max_h, max_w = 256, 512
        det2mask = Det2Mask()
        boxes, scores, classes, _ = generate_boxes(
            max_h, max_w, num_boxes, batch_size
        )
        masks, scores, class_ids = det2mask(
            [torch.rand(num_boxes, num_classes, 28, 28).sigmoid()]
            * batch_size,
            boxes,
            scores,
            classes,
            [(max_h, max_w)] * batch_size,
        )
        assert len(masks) == len(scores) == len(class_ids) == batch_size
        for j in range(batch_size):
            assert (
                masks[j].shape[0]
                == scores[j].shape[0]
                == class_ids[j].shape[0]
                == num_boxes
            )
            assert masks[j].shape[1] == max_h
            assert masks[j].shape[2] == max_w
