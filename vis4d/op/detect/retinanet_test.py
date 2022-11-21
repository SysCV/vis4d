"""RetinaNet tests."""
import unittest

import torch

from vis4d.op.box.encoder import DeltaXYWHBBoxEncoder

from .retinanet import Dense2Det, RetinaNetHead, get_default_anchor_generator


class RetinaNetTest(unittest.TestCase):
    """RetinaNet test class."""

    def test_retinanet_head(self):
        """Test RetinaNetHead class."""
        batch_size, num_classes, wh, inc = 2, 5, 128, 64
        # default setup
        retinanet_head = RetinaNetHead(num_classes, inc, feat_channels=64)
        test_features = [None, None] + [
            torch.rand(batch_size, inc, wh // 2**i, wh // 2**i)
            for i in range(5)
        ]
        cls_score, bbox_pred = retinanet_head(test_features[2:])
        assert len(cls_score) == len(bbox_pred) == 5
        for j, (score, box) in enumerate(zip(cls_score, bbox_pred)):
            assert len(score) == len(box) == batch_size
            wh_ = wh // 2**j
            assert score.shape[2:] == box.shape[2:] == (wh_, wh_)
            assert score.shape[1] == 9 * num_classes
            assert box.shape[1] == 9 * 4

    def test_dense2det(self):
        """Test Dense2Det class."""
        # default setup
        batch_size, num_classes, wh = 2, 5, 128
        max_h, max_w = wh * 4, wh * 4
        dense2det = Dense2Det(
            get_default_anchor_generator(), DeltaXYWHBBoxEncoder()
        )
        test_cls = [
            torch.rand(batch_size, num_classes * 9, wh // 2**i, wh // 2**i)
            for i in range(5)
        ]
        test_reg = [
            torch.rand(batch_size, 4 * 9, wh // 2**i, wh // 2**i)
            for i in range(5)
        ]
        boxes, scores, class_ids = dense2det(
            test_cls, test_reg, [(max_h, max_w)] * batch_size
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
