"""Test cases for Vis4D models."""
import unittest
from typing import Optional

from vis4d.common.bbox.poolers import MultiScaleRoIAlign
from vis4d.common.bbox.matchers import MaxIoUMatcher
from vis4d.common.bbox.samplers import CombinedSampler
from vis4d.model.detect import (
    D2TwoStageDetector,
    MMOneStageDetector,
    MMTwoStageDetector,
)
from vis4d.model.heads.dense_head import MMSegDecodeHead
from vis4d.model.heads.panoptic_head import SimplePanopticHead
from vis4d.model.heads.roi_head import QD3DTBBox3DHead
from vis4d.model.panoptic import PanopticFPN
from vis4d.model.segment import MMEncDecSegmentor
from vis4d.model.track.graph import QDTrackGraph
from vis4d.model.track.similarity import QDSimilarityHead
from vis4d.unittest.utils import generate_input_sample

from .base import BaseModel
from .qdtrack import QDTrack
from .qd_3dt import QD3DT


class BaseModelTests:
    """Base class for model tests."""

    class TestDetect(unittest.TestCase):
        """Base test case for vis4d detect models."""

        model: Optional[BaseModel] = None
        category_mapping = {"pedestrian": 0, "rider": 1, "car": 2}

        def test_train(self) -> None:
            """Test case for training."""
            assert self.model is not None
            self.model.training = True
            inputs = [generate_input_sample(32, 32, 2, 3, use_score=False)]
            outs = self.model(inputs)
            self.assertTrue(isinstance(outs, dict))
            self.assertGreater(len(outs), 0)

        def test_test(self) -> None:
            """Test case for testing."""
            assert self.model is not None
            self.model.training = False
            inputs = [generate_input_sample(32, 32, 1, 3, use_score=False)]
            outs = self.model(inputs)
            self.assertTrue(isinstance(outs, dict))
            self.assertGreater(len(outs), 0)

    class TestDetect3D(unittest.TestCase):
        """Base test case for vis4d 3D detect models."""

        model: Optional[BaseModel] = None
        category_mapping = {"pedestrian": 0, "rider": 1, "car": 2}

        def test_train(self) -> None:
            """Test case for training."""
            assert self.model is not None
            self.model.training = True
            inputs = [
                generate_input_sample(
                    32, 32, 2, 3, det3d_input=True, use_score=False
                )
            ]
            outs = self.model(inputs)
            self.assertTrue(isinstance(outs, dict))
            self.assertGreater(len(outs), 0)

        def test_test(self) -> None:
            """Test case for testing."""
            assert self.model is not None
            self.model.training = False
            inputs = [
                generate_input_sample(
                    32, 32, 1, 3, det3d_input=True, use_score=False
                )
            ]
            outs = self.model(inputs)
            self.assertTrue(isinstance(outs, dict))
            self.assertGreater(len(outs), 0)

    class TestSegment(unittest.TestCase):
        """Base test case for vis4d segment models."""

        model: Optional[BaseModel] = None
        category_mapping = {"pedestrian": 0, "rider": 1, "car": 2}

        def test_train(self) -> None:
            """Test case for training."""
            assert self.model is not None
            self.model.training = True
            inputs = [generate_input_sample(32, 32, 2, 3, det_input=False)]
            outs = self.model(inputs)
            self.assertTrue(isinstance(outs, dict))
            self.assertGreater(len(outs), 0)

        def test_test(self) -> None:
            """Test case for testing."""
            assert self.model is not None
            self.model.training = False
            inputs = [generate_input_sample(32, 32, 2, 3, det_input=False)]
            outs = self.model(inputs)
            self.assertTrue(isinstance(outs, dict))
            self.assertGreater(len(outs), 0)

    class TestPanoptic(unittest.TestCase):
        """Base test case for vis4d panoptic models."""

        model: Optional[BaseModel] = None
        category_mapping = {"pedestrian": 0, "rider": 1, "car": 2}

        def test_train(self) -> None:
            """Test case for training."""
            assert self.model is not None
            self.model.training = True
            inputs = [
                generate_input_sample(
                    32, 32, 2, 3, pan_input=True, use_score=False
                )
            ]
            outs = self.model(inputs)
            self.assertTrue(isinstance(outs, dict))
            self.assertGreater(len(outs), 0)

        def test_test(self) -> None:
            """Test case for testing."""
            assert self.model is not None
            self.model.training = False
            inputs = [
                generate_input_sample(
                    32, 32, 2, 3, pan_input=True, use_score=False
                )
            ]
            outs = self.model(inputs)
            self.assertTrue(isinstance(outs, dict))
            self.assertGreater(len(outs), 0)


class TestDetectMMFasterRCNN(BaseModelTests.TestDetect):
    """MMDetection Faster R-CNN detection test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = MMTwoStageDetector(
            model_base="vis4d/model/testcases/mmdet_cfg_frcnn_r18fpn.py",
            pixel_mean=(123.675, 116.28, 103.53),
            pixel_std=(58.395, 57.12, 57.375),
            model_kwargs={
                "rpn_head.loss_bbox.type": "SmoothL1Loss",
                "rpn_head.loss_bbox.beta": 0.111,
                "roi_head.bbox_head.loss_bbox.type": "SmoothL1Loss",
            },
            category_mapping=cls.category_mapping,
        )


class TestDetectMMMaskRCNN(BaseModelTests.TestDetect):
    """MMDetection Mask R-CNN detection test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = MMTwoStageDetector(
            model_base="mmdet://_base_/models/mask_rcnn_r50_fpn.py",
            pixel_mean=(123.675, 116.28, 103.53),
            pixel_std=(58.395, 57.12, 57.375),
            weights="mmdet://faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth",  # pylint: disable=line-too-long
            category_mapping=cls.category_mapping,
        )


class TestDetectMMRetinaNet(BaseModelTests.TestDetect):
    """MMDetection RetinaNet detection test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = MMOneStageDetector(
            model_base="mmdet://_base_/models/retinanet_r50_fpn.py",
            pixel_mean=(123.675, 116.28, 103.53),
            pixel_std=(58.395, 57.12, 57.375),
            weights="bdd100k://det/models/retinanet_r50_fpn_3x_det_bdd100k.pth",  # pylint: disable=line-too-long
            category_mapping=cls.category_mapping,
        )


class TestDetectD2MaskRCNN(BaseModelTests.TestDetect):
    """Detectron2 Mask R-CNN detection test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = D2TwoStageDetector(
            model_base="mask-rcnn/r50-fpn",
            pixel_mean=(123.675, 116.28, 103.53),
            pixel_std=(58.395, 57.12, 57.375),
            image_channel_mode="BGR",
            weights="detectron2",
            category_mapping=cls.category_mapping,
        )


class TestQDTrackFasterRCNN(BaseModelTests.TestDetect):
    """QDTrack with Faster R-CNN track test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = QDTrack(
            detection=MMTwoStageDetector(
                model_base="vis4d/model/testcases/mmdet_cfg_frcnn_r18fpn.py",
                pixel_mean=(123.675, 116.28, 103.53),
                pixel_std=(58.395, 57.12, 57.375),
                backbone_output_names=["p2", "p3", "p4", "p5", "p6"],
                category_mapping=cls.category_mapping,
            ),
            similarity=QDSimilarityHead(in_dim=64),
            track_graph=QDTrackGraph(10),
            category_mapping=cls.category_mapping,
        )


class TestQDTrackRetinaNet(BaseModelTests.TestDetect):
    """QDTrack with RetinaNet track test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = QDTrack(
            detection=MMOneStageDetector(
                model_base="vis4d/model/testcases/mmdet_cfg_retinanet_r18fpn.py",  # pylint: disable=line-too-long
                pixel_mean=(123.675, 116.28, 103.53),
                pixel_std=(58.395, 57.12, 57.375),
                backbone_output_names=["p2", "p3", "p4", "p5", "p6"],
                category_mapping=cls.category_mapping,
            ),
            similarity=QDSimilarityHead(in_dim=64),
            track_graph=QDTrackGraph(10),
            category_mapping=cls.category_mapping,
        )


class TestQD3DT(BaseModelTests.TestDetect3D):
    """QD3DT test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = QD3DT(
            detection=MMTwoStageDetector(
                model_base="vis4d/model/testcases/mmdet_cfg_frcnn_r18fpn.py",
                pixel_mean=(123.675, 116.28, 103.53),
                pixel_std=(58.395, 57.12, 57.375),
                model_kwargs={
                    "rpn_head.anchor_generator.scales": [4, 8],
                    "rpn_head.anchor_generator.ratios": [
                        0.25,
                        0.5,
                        1.0,
                        2.0,
                        4.0,
                    ],
                    "rpn_head.loss_bbox.type": "SmoothL1Loss",
                    "rpn_head.loss_bbox.beta": 0.111,
                    "roi_head.bbox_head.type": "ConvFCBBoxHead",
                    "roi_head.bbox_head.num_shared_convs": 4,
                    "roi_head.bbox_head.num_shared_fcs": 2,
                    "roi_head.bbox_head.loss_cls.loss_weight": 5.0,
                    "roi_head.bbox_head.loss_bbox.type": "SmoothL1Loss",
                    "roi_head.bbox_head.loss_bbox.beta": 0.111,
                    "roi_head.bbox_head.loss_bbox.loss_weight": 5.0,
                },
                backbone_output_names=["p2", "p3", "p4", "p5", "p6"],
                category_mapping=cls.category_mapping,
            ),
            bbox_3d_head=QD3DTBBox3DHead(
                num_classes=3,
                proposal_pooler=MultiScaleRoIAlign(
                    sampling_ratio=0, resolution=(7, 7), strides=[4, 8, 16, 32]
                ),
                proposal_sampler=CombinedSampler(
                    batch_size_per_image=64,
                    positive_fraction=0.25,
                    pos_strategy="instance_balanced",
                    neg_strategy="iou_balanced",
                ),
                proposal_matcher=MaxIoUMatcher(
                    thresholds=[0.5, 0.5],
                    labels=[0, -1, 1],
                    allow_low_quality_matches=False,
                ),
                in_channels=64,
                fc_out_dim=64,
                roi_feat_size=7,
                conv_has_bias=False,
                proposal_append_gt=True,
            ),
            similarity=QDSimilarityHead(in_dim=64),
            track_graph=QDTrackGraph(10),
            image_channel_mode="RGB",
            category_mapping=cls.category_mapping,
        )


class TestSegmentMMFPN(BaseModelTests.TestSegment):
    """MMSegmentation FPN segmentation test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = MMEncDecSegmentor(
            model_base="vis4d/model/testcases/mmseg_cfg_fpn_r18.py",
            pixel_mean=(123.675, 116.28, 103.53),
            pixel_std=(58.395, 57.12, 57.375),
            category_mapping=cls.category_mapping,
        )


class TestSegmentMMDeepLab(BaseModelTests.TestSegment):
    """MMSegmentation DeepLab segmentation test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = MMEncDecSegmentor(
            model_base="mmseg://_base_/models/deeplabv3_r50-d8.py",
            pixel_mean=(123.675, 116.28, 103.53),
            pixel_std=(58.395, 57.12, 57.375),
            model_kwargs={
                "pretrained": "open-mmlab://resnet18_v1c",
                "backbone.depth": 18,
                "backbone.norm_cfg.type": "BN",
                "decode_head.norm_cfg.type": "BN",
                "decode_head.in_channels": 512,
                "decode_head.channels": 128,
                "auxiliary_head.norm_cfg.type": "BN",
                "auxiliary_head.in_channels": 256,
                "auxiliary_head.channels": 64,
            },
            weights="mmseg://deeplabv3/deeplabv3_r18-d8_512x1024_80k_cityscapes/deeplabv3_r18-d8_512x1024_80k_cityscapes_20201225_021506-23dffbe2.pth",  # pylint: disable=line-too-long
            category_mapping=cls.category_mapping,
        )


class TestPanopticFPN(BaseModelTests.TestPanoptic):
    """Panoptic FPN test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = PanopticFPN(
            detection=MMTwoStageDetector(
                model_base="vis4d/model/testcases/mmdet_cfg_maskrcnn_r18fpn.py",  # pylint: disable=line-too-long
                pixel_mean=(123.675, 116.28, 103.53),
                pixel_std=(58.395, 57.12, 57.375),
                backbone_output_names=["p2", "p3", "p4", "p5", "p6"],
                category_mapping=cls.category_mapping,
            ),
            seg_head=MMSegDecodeHead(
                mm_cfg="vis4d/model/testcases/fpn_seg_head.py",
                category_mapping=cls.category_mapping,
            ),
            pan_head=SimplePanopticHead(ignore_class=2),
            category_mapping=cls.category_mapping,
        )
