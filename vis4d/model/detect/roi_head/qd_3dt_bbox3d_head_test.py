"""Testcases for roi head."""
import unittest

import torch
from scalabel.label.typing import Frame

from vis4d.common.bbox.coders import BaseBoxCoderConfig
from vis4d.common.bbox.matchers import MatcherConfig
from vis4d.common.bbox.poolers import RoIPoolerConfig
from vis4d.common.bbox.samplers import SamplerConfig
from vis4d.model.losses import LossConfig
from vis4d.struct import Images, InputSample
from vis4d.unittest.utils import generate_dets, generate_feature_list

from .base import BaseRoIHeadConfig
from .qd_3dt_bbox3d_head import QD3DTBBox3DHead


class TestQDTBBox3DHead(unittest.TestCase):
    """Testcases for 3D head in QD-3DT."""

    def test_box3d_detection(self) -> None:
        """Testcase for box3d detection."""
        boxcoder_cfg = BaseBoxCoderConfig(type="QD3DTBox3DCoder")
        loss_cfg = LossConfig(type="Box3DUncertaintyLoss")

        matcher_cfg = MatcherConfig(
            type="MaxIoUMatcher",
            thresholds=[0.5, 0.5],
            labels=[0, -1, 1],
            allow_low_quality_matches=False,
        )
        pooler_cfg = RoIPoolerConfig(
            type="MultiScaleRoIPooler",
            pooling_op="RoIAlign",
            resolution=[7, 7],
            strides=[4, 8, 16, 32],
            sampling_ratio=0,
        )
        sampler_cfg = SamplerConfig(
            type="CombinedSampler",
            batch_size_per_image=512,
            positive_fraction=0.25,
            pos_strategy="instance_balanced",
            neg_strategy="iou_balanced",
        )

        box3d_head = QD3DTBBox3DHead(
            BaseRoIHeadConfig(
                type="QD3DT",
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
        )

        c, h, w, list_len = 256, 128, 128, 4

        detections = generate_dets(h, w, 1)
        features_list = generate_feature_list(c, h, w, list_len)

        boxes_3d_pred = box3d_head(features_list, [detections])

        self.assertTrue(len(boxes_3d_pred) == len(detections))

        # Test no detections corner case
        detections = generate_dets(h, w, 0)

        inputs = InputSample(
            [Frame(name="test")], Images(torch.rand(1, 3, 1, 1), [(1, 1)])
        )

        boxes_3d_pred = box3d_head.forward_test(
            inputs, [detections], {"test": torch.rand(1)}
        )

        self.assertTrue(len(boxes_3d_pred[0]) == len(detections) == 0)
