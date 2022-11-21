"""RetinaNet tests."""
import unittest

import torch

from .retinanet import RetinaNetHead


class RetinaNetTest(unittest.TestCase):
    """RetinaNet test class."""

    def test_retinanet_head(self):
        """Test RetinaNetHead class."""
        batch_size, num_classes, wh, inc = 2, 5, 128, 64
        # default setup
        retinanet_head = RetinaNetHead(num_classes, inc, feat_channels=64)
        test_features = [None, None] + [
            torch.rand(batch_size, inc, wh // 2 ** i, wh // 2 ** i)
            for i in range(5)
        ]
        cls_score, bbox_pred = retinanet_head(test_features[2:])
        assert len(cls_score) == len(bbox_pred) == 5
        for j, (score, box) in enumerate(zip(cls_score, bbox_pred)):
            assert len(score) == len(box) == batch_size
            wh_ = wh // 2 ** j
            assert score.shape[2:] == box.shape[2:] == (wh_, wh_)
            assert score.shape[1] == 9 * num_classes
            assert box.shape[1] == 9 * 4
