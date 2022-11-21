"""Faster RCNN tests."""
import unittest

from vis4d.unittest.util import generate_boxes, generate_features

from .faster_rcnn import FasterRCNNHead


class FasterRCNNTest(unittest.TestCase):
    """Faster RCNN test class."""

    def test_faster_rcnn_head(self):
        """Test FasterRCNNHead class."""
        batch_size, num_classes, num_boxes, wh = 2, 5, 10, 128
        # default setup
        faster_rcnn_head = FasterRCNNHead(num_classes)
        test_features = [None, None] + generate_features(
            256, wh, wh, 5, batch_size
        )
        # train forward
        boxes, _, classes, _ = generate_boxes(
            wh * 4, wh * 4, num_boxes, batch_size
        )
        rpn, roi, props, smp_props, smp_tgts, smp_tgt_inds = faster_rcnn_head(
            test_features, [(wh * 4, wh * 4)] * batch_size, boxes, classes
        )
        rpn_cls, rpn_box = rpn
        assert len(rpn_cls) == len(rpn_box) == 5  # number of pyramid levels
        for j, (rpnc, rpnb) in enumerate(zip(rpn_cls, rpn_box)):
            assert len(rpnc) == len(rpnb) == batch_size
            wh_ = wh // 2**j
            assert rpnc.shape[2:] == rpnb.shape[2:] == (wh_, wh_)
        cls_score, bbox_pred = roi
        assert cls_score.shape == (batch_size * 512, num_classes + 1)
        assert bbox_pred.shape == (batch_size * 512, num_classes * 4)
        boxes, scores = props
        assert len(boxes) == len(scores) == batch_size
        for j, (box, score) in enumerate(zip(boxes, scores)):
            assert len(box) == len(score) == 1000
            assert box.shape[1] == 4
            assert score.dim() == 1
        for smp in (smp_props, smp_tgts, smp_tgt_inds):
            assert smp is not None
        # test forward
        rpn, roi, props, smp_props, smp_tgts, smp_tgt_inds = faster_rcnn_head(
            test_features, [(wh * 4, wh * 4)] * batch_size
        )
        assert smp_props == smp_tgts == smp_tgt_inds == None
        cls_score, bbox_pred = roi
        assert cls_score.shape == (batch_size * 1000, num_classes + 1)
        assert bbox_pred.shape == (batch_size * 1000, num_classes * 4)
        boxes, scores = props
        assert len(boxes) == len(scores) == batch_size
        for j, (box, score) in enumerate(zip(boxes, scores)):
            assert len(box) == len(score) == 1000
            assert box.shape[1] == 4
            assert score.dim() == 1
