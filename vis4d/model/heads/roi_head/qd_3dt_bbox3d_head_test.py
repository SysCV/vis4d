"""Testcases for roi head."""
import unittest

import torch
from scalabel.label.typing import Frame

from vis4d.common.bbox.coders import QD3DTBox3DCoder
from vis4d.common.bbox.matchers import MaxIoUMatcher
from vis4d.common.bbox.poolers import MultiScaleRoIPooler
from vis4d.common.bbox.samplers import CombinedSampler
from vis4d.model.losses import Box3DUncertaintyLoss
from vis4d.struct import Images, InputSample
from vis4d.unittest.utils import generate_dets, generate_feature_list

from .qd_3dt_bbox3d_head import QD3DTBBox3DHead


class TestQDTBBox3DHead(unittest.TestCase):
    """Testcases for 3D head in QD-3DT."""

    def test_box3d_detection(self) -> None:
        """Testcase for box3d detection."""
        boxcoder_cfg = QD3DTBox3DCoder()
        loss_cfg = Box3DUncertaintyLoss()

        matcher_cfg = MaxIoUMatcher(
            thresholds=[0.5, 0.5],
            labels=[0, -1, 1],
            allow_low_quality_matches=False,
        )
        pooler_cfg = MultiScaleRoIPooler(
            pooling_op="RoIAlign",
            resolution=(7, 7),
            strides=[4, 8, 16, 32],
            sampling_ratio=0,
        )
        sampler_cfg = CombinedSampler(
            batch_size_per_image=512,
            positive_fraction=0.25,
            pos_strategy="instance_balanced",
            neg_strategy="iou_balanced",
        )

        box3d_head = QD3DTBBox3DHead(
            num_classes=1,
            num_shared_fcs=1,
            num_dep_convs=0,
            num_dep_fcs=1,
            num_dim_convs=0,
            num_dim_fcs=1,
            num_rot_convs=0,
            num_rot_fcs=1,
            num_2dc_convs=0,
            num_2dc_fcs=1,
            norm="GroupNorm",
            proposal_append_gt=True,
            loss=loss_cfg,
            box3d_coder=boxcoder_cfg,
            proposal_pooler=pooler_cfg,
            proposal_sampler=sampler_cfg,
            proposal_matcher=matcher_cfg,
        )

        c, h, w, list_len = 256, 128, 128, 4

        detections = generate_dets(h, w, 1)
        features_list = generate_feature_list(c, h, w, list_len)

        boxes_3d_pred = box3d_head.get_predictions(features_list, [detections])

        self.assertTrue(len(boxes_3d_pred) == len(detections))

        # Test no detections corner case
        detections = generate_dets(h, w, 0)

        inputs = InputSample(
            [Frame(name="test")], Images(torch.rand(1, 3, 1, 1), [(1, 1)])
        )

        boxes_3d_pred = box3d_head(
            inputs, [detections], {"test": torch.rand(1)}
        )

        # pylint: disable=unsubscriptable-object
        self.assertTrue(len(boxes_3d_pred[0]) == len(detections) == 0)
