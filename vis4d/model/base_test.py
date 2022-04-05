"""Test cases for Vis4D models."""
import unittest
from typing import Optional

import torch

from vis4d.common.bbox.matchers import MaxIoUMatcher
from vis4d.common.bbox.poolers import MultiScaleRoIAlign
from vis4d.common.bbox.samplers import CombinedSampler
from vis4d.model.backbone import MMDetBackbone
from vis4d.model.backbone.neck import MMDetNeck
from vis4d.model.detect import (
    D2TwoStageDetector,
    MMOneStageDetector,
    MMTwoStageDetector,
)
from vis4d.model.heads.dense_head import (
    MMDetDenseHead,
    MMDetRPNHead,
    MMSegDecodeHead,
)
from vis4d.model.heads.panoptic_head import SimplePanopticHead
from vis4d.model.heads.roi_head import MMDetRoIHead, QD3DTBBox3DHead
from vis4d.model.panoptic import PanopticFPN
from vis4d.model.segment import MMEncDecSegmentor
from vis4d.model.track.graph import QDTrackGraph
from vis4d.model.track.similarity import QDSimilarityHead
from vis4d.unittest.utils import generate_input_sample

from .base import BaseModel
from .qd_3dt import QD3DT
from .qdtrack import QDTrack

PIXEL_MEAN = (123.675, 116.28, 103.53)
PIXEL_STD = (58.395, 57.12, 57.375)
TEST_MAPPING = {"pedestrian": 0, "rider": 1, "car": 2}


class BaseModelTests:
    """Base class for model tests."""

    class TestDetect(unittest.TestCase):
        """Base test case for vis4d detect models."""

        model: Optional[BaseModel] = None

        def test_train(self) -> None:
            """Test case for training."""
            assert self.model is not None
            self.model.train()
            inputs = [generate_input_sample(32, 32, 2, 3, use_score=False)]
            outs = self.model(inputs)
            self.assertTrue(isinstance(outs, dict))
            self.assertGreater(len(outs), 0)

        def test_test(self) -> None:
            """Test case for testing."""
            assert self.model is not None
            self.model.eval()
            inputs = [generate_input_sample(32, 32, 1, 3, use_score=False)]
            outs = self.model(inputs)
            self.assertTrue(isinstance(outs, dict))
            self.assertGreater(len(outs), 0)

    class TestTrack(unittest.TestCase):
        """Base test case for vis4d tracking models."""

        model: Optional[BaseModel] = None

        def test_train(self) -> None:
            """Test case for training."""
            assert self.model is not None
            self.model.train()
            inputs = [
                generate_input_sample(
                    32, 32, 2, 3, use_score=False, track_ids=True
                ),
                generate_input_sample(
                    32, 32, 2, 3, use_score=False, track_ids=True
                ),
            ]
            outs = self.model(inputs)
            self.assertTrue(isinstance(outs, dict))
            self.assertGreater(len(outs), 0)

        def test_test(self) -> None:
            """Test case for testing."""
            assert self.model is not None
            with torch.no_grad():
                self.model.eval()
                inputs = [generate_input_sample(32, 32, 1, 3, use_score=False)]
                outs = self.model(inputs)
                self.assertTrue(isinstance(outs, dict))
                self.assertGreater(len(outs), 0)

    class TestTrack3D(unittest.TestCase):
        """Base test case for vis4d 3D track models."""

        model: Optional[BaseModel] = None

        def test_train(self) -> None:
            """Test case for training."""
            assert self.model is not None
            self.model.train()
            inputs = [
                generate_input_sample(
                    32,
                    32,
                    2,
                    3,
                    det3d_input=True,
                    use_score=False,
                    track_ids=True,
                ),
                generate_input_sample(
                    32,
                    32,
                    2,
                    3,
                    det3d_input=True,
                    use_score=False,
                    track_ids=True,
                ),
            ]
            outs = self.model(inputs)
            self.assertTrue(isinstance(outs, dict))
            self.assertGreater(len(outs), 0)

        def test_test(self) -> None:
            """Test case for testing."""
            assert self.model is not None
            with torch.no_grad():
                self.model.eval()
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

        def test_train(self) -> None:
            """Test case for training."""
            assert self.model is not None
            self.model.train()
            inputs = [generate_input_sample(32, 32, 2, 3, det_input=False)]
            outs = self.model(inputs)
            self.assertTrue(isinstance(outs, dict))
            self.assertGreater(len(outs), 0)

        def test_test(self) -> None:
            """Test case for testing."""
            assert self.model is not None
            with torch.no_grad():
                self.model.eval()
                inputs = [generate_input_sample(32, 32, 2, 3, det_input=False)]
                outs = self.model(inputs)
                self.assertTrue(isinstance(outs, dict))
                self.assertGreater(len(outs), 0)

    class TestPanoptic(unittest.TestCase):
        """Base test case for vis4d panoptic models."""

        model: Optional[BaseModel] = None

        def test_train(self) -> None:
            """Test case for training."""
            assert self.model is not None
            self.model.train()
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
            self.model.eval()
            with torch.no_grad():
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
            "extra_args_test",
            model_base="vis4d/model/testcases/mmdet_cfg_frcnn_r18fpn.py",
            pixel_mean=PIXEL_MEAN,
            pixel_std=PIXEL_STD,
            model_kwargs={
                "rpn_head.loss_bbox.type": "SmoothL1Loss",
                "rpn_head.loss_bbox.beta": 0.111,
                "roi_head.bbox_head.loss_bbox.type": "SmoothL1Loss",
            },
            category_mapping=TEST_MAPPING,
            extra_kwargs_test=None,
        )


class TestDetectMMMaskRCNN(BaseModelTests.TestDetect):
    """MMDetection Mask R-CNN detection test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = MMTwoStageDetector(
            model_base="mmdet://_base_/models/mask_rcnn_r50_fpn.py",
            pixel_mean=PIXEL_MEAN,
            pixel_std=PIXEL_STD,
            weights="mmdet://faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth",  # pylint: disable=line-too-long
            category_mapping=TEST_MAPPING,
        )


class TestDetectMMRetinaNet(BaseModelTests.TestDetect):
    """MMDetection RetinaNet detection test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = MMOneStageDetector(
            model_base="mmdet://_base_/models/retinanet_r50_fpn.py",
            pixel_mean=PIXEL_MEAN,
            pixel_std=PIXEL_STD,
            weights="bdd100k://det/models/retinanet_r50_fpn_3x_det_bdd100k.pth",  # pylint: disable=line-too-long
            category_mapping=TEST_MAPPING,
        )


class TestDetectD2MaskRCNN(BaseModelTests.TestDetect):
    """Detectron2 Mask R-CNN detection test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = D2TwoStageDetector(
            model_base="mask-rcnn/r50-fpn",
            pixel_mean=PIXEL_MEAN,
            pixel_std=PIXEL_STD,
            image_channel_mode="BGR",
            weights="detectron2",
            category_mapping=TEST_MAPPING,
            set_batchnorm_eval=True,
        )


class TestQDTrackMaskRCNN(BaseModelTests.TestTrack):
    """QDTrack with Mask R-CNN track test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = QDTrack(
            detection=MMTwoStageDetector(
                model_base="vis4d/model/testcases/mmdet_cfg_maskrcnn_r18fpn.py",  # pylint: disable=line-too-long
                pixel_mean=PIXEL_MEAN,
                pixel_std=PIXEL_STD,
                backbone_output_names=["p2", "p3", "p4", "p5", "p6"],
                category_mapping=TEST_MAPPING,
            ),
            similarity=QDSimilarityHead(in_dim=64),
            track_graph=QDTrackGraph(10),
            category_mapping=TEST_MAPPING,
        )


class TestQDTrackInferenceResults(BaseModelTests.TestTrack):
    """QDTrack with Faster R-CNN track with inference results test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = QDTrack(
            detection=MMTwoStageDetector(
                model_base="vis4d/model/testcases/mmdet_cfg_frcnn_r18fpn.py",
                pixel_mean=PIXEL_MEAN,
                pixel_std=PIXEL_STD,
                backbone_output_names=["p2", "p3", "p4", "p5", "p6"],
                category_mapping=TEST_MAPPING,
            ),
            similarity=QDSimilarityHead(
                in_dim=64,
                proposal_pooler=MultiScaleRoIAlign(
                    resolution=[7, 7], strides=[4, 8, 16, 32], sampling_ratio=0
                ),
                proposal_sampler=CombinedSampler(
                    batch_size_per_image=256,
                    positive_fraction=0.5,
                    pos_strategy="instance_balanced",
                    neg_strategy="iou_balanced",
                ),
                proposal_matcher=MaxIoUMatcher(
                    thresholds=[0.3, 0.7],
                    labels=[0, -1, 1],
                    allow_low_quality_matches=False,
                ),
            ),
            track_graph=QDTrackGraph(10),
            category_mapping=TEST_MAPPING,
            inference_result_path="unittests/results.hdf5",
        )


class TestQDTrackRetinaNet(BaseModelTests.TestTrack):
    """QDTrack with RetinaNet track test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = QDTrack(
            detection=MMOneStageDetector(
                model_base="vis4d/model/testcases/mmdet_cfg_retinanet_r18fpn.py",  # pylint: disable=line-too-long
                pixel_mean=PIXEL_MEAN,
                pixel_std=PIXEL_STD,
                backbone_output_names=["p2", "p3", "p4", "p5", "p6"],
                category_mapping=TEST_MAPPING,
            ),
            similarity=QDSimilarityHead(in_dim=64),
            track_graph=QDTrackGraph(10),
            category_mapping=TEST_MAPPING,
        )


class TestQD3DT(BaseModelTests.TestTrack3D):
    """QD3DT test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = QD3DT(
            detection=MMTwoStageDetector(
                model_base="vis4d/model/testcases/mmdet_cfg_frcnn_r18fpn.py",
                pixel_mean=PIXEL_MEAN,
                pixel_std=PIXEL_STD,
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
                category_mapping=TEST_MAPPING,
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
            category_mapping=TEST_MAPPING,
        )


class TestSegmentMMFPN(BaseModelTests.TestSegment):
    """MMSegmentation FPN segmentation test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = MMEncDecSegmentor(
            model_base="vis4d/model/testcases/mmseg_cfg_fpn_r18.py",
            pixel_mean=PIXEL_MEAN,
            pixel_std=PIXEL_STD,
            category_mapping=TEST_MAPPING,
        )


class TestSegmentMMDeepLab(BaseModelTests.TestSegment):
    """MMSegmentation DeepLab segmentation test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = MMEncDecSegmentor(
            model_base="mmseg://_base_/models/deeplabv3_r50-d8.py",
            pixel_mean=PIXEL_MEAN,
            pixel_std=PIXEL_STD,
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
            category_mapping=TEST_MAPPING,
        )


class TestPanopticFPN(BaseModelTests.TestPanoptic):
    """Panoptic FPN test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        cls.model = PanopticFPN(
            detection=MMTwoStageDetector(
                model_base="vis4d/model/testcases/mmdet_cfg_maskrcnn_r18fpn.py",  # pylint: disable=line-too-long
                pixel_mean=PIXEL_MEAN,
                pixel_std=PIXEL_STD,
                backbone_output_names=["p2", "p3", "p4", "p5", "p6"],
                category_mapping=TEST_MAPPING,
            ),
            seg_head=MMSegDecodeHead(
                mm_cfg="vis4d/model/testcases/fpn_seg_head.py",
                category_mapping=TEST_MAPPING,
            ),
            pan_head=SimplePanopticHead(ignore_class=2),
            category_mapping=TEST_MAPPING,
        )


class TestBaseModel(unittest.TestCase):
    """Test cases for BaseModel."""

    def test_load_weights_and_freeze(self) -> None:
        """Test loading pretrained weights and freezing params."""
        model = MMOneStageDetector(
            model_base="mmdet://_base_/models/retinanet_r50_fpn.py",
            pixel_mean=PIXEL_MEAN,
            pixel_std=PIXEL_STD,
            category_mapping=TEST_MAPPING,
            strict=False,
            freeze=True,
            freeze_parameters=["bbox_head"],
        )
        model._weights = (  # pylint: disable=protected-access
            "https://dl.cv.ethz.ch/bdd100k/"
            + "det/models/retinanet_r50_fpn_3x_det_bdd100k.pth"
        )
        model._revise_keys = [  # pylint: disable=protected-access
            (r"^bbox_head\.", "bbox_head.mm_dense_head."),
            (r"^backbone\.", "backbone.mm_backbone."),
            (r"^neck\.", "backbone.neck.mm_neck."),
        ]
        model.on_fit_start()
        self.assertTrue(model._freeze)  # pylint: disable=protected-access


class TestModelConstruction(unittest.TestCase):
    """Test cases for constructing models."""

    def test_two_stage_detector(self) -> None:
        """Two stage detector test case."""
        model = MMTwoStageDetector(
            backbone=MMDetBackbone(
                mm_cfg=dict(
                    type="ResNet",
                    depth=50,
                    num_stages=4,
                    out_indices=(0, 1, 2, 3),
                    norm_cfg=dict(type="BN", requires_grad=True),
                    style="pytorch",
                    init_cfg=dict(
                        type="Pretrained", checkpoint="torchvision://resnet50"
                    ),
                ),
                pixel_mean=PIXEL_MEAN,
                pixel_std=PIXEL_STD,
                out_indices=[0, 1, 2, 3],
                neck=MMDetNeck(
                    mm_cfg=dict(
                        type="FPN",
                        in_channels=[256, 512, 1024, 2048],
                        out_channels=256,
                        num_outs=5,
                    ),
                ),
            ),
            rpn_head=MMDetRPNHead(
                mm_cfg=dict(
                    type="RPNHead",
                    in_channels=256,
                    feat_channels=256,
                    anchor_generator=dict(
                        type="AnchorGenerator",
                        scales=[8],
                        ratios=[0.5, 1.0, 2.0],
                        strides=[4, 8, 16, 32, 64],
                    ),
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[1.0, 1.0, 1.0, 1.0],
                    ),
                    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True),
                    loss_bbox=dict(type="L1Loss", loss_weight=1.0),
                    train_cfg=dict(
                        assigner=dict(
                            type="MaxIoUAssigner",
                            pos_iou_thr=0.7,
                            neg_iou_thr=0.3,
                            min_pos_iou=0.3,
                            match_low_quality=True,
                            ignore_iof_thr=-1,
                        ),
                        sampler=dict(
                            type="RandomSampler",
                            num=256,
                            pos_fraction=0.5,
                            add_gt_as_proposals=False,
                        ),
                        rpn_proposal=dict(
                            nms_pre=2000,
                            max_per_img=1000,
                            nms=dict(type="nms", iou_threshold=0.7),
                            min_bbox_size=0,
                        ),
                        allowed_border=-1,
                        pos_weight=-1,
                    ),
                ),
                category_mapping=TEST_MAPPING,
            ),
            roi_head=MMDetRoIHead(
                mm_cfg=dict(
                    type="StandardRoIHead",
                    bbox_roi_extractor=dict(
                        type="SingleRoIExtractor",
                        roi_layer=dict(
                            type="RoIAlign", output_size=7, sampling_ratio=0
                        ),
                        out_channels=256,
                        featmap_strides=[4, 8, 16, 32],
                    ),
                    bbox_head=dict(
                        type="Shared2FCBBoxHead",
                        in_channels=256,
                        roi_feat_size=7,
                        num_classes=80,
                        bbox_coder=dict(
                            type="DeltaXYWHBBoxCoder",
                            target_means=[0.0, 0.0, 0.0, 0.0],
                            target_stds=[0.1, 0.1, 0.2, 0.2],
                        ),
                        reg_class_agnostic=False,
                        loss_cls=dict(
                            type="CrossEntropyLoss", use_sigmoid=False
                        ),
                        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
                    ),
                    train_cfg=dict(
                        assigner=dict(
                            type="MaxIoUAssigner",
                            pos_iou_thr=0.5,
                            neg_iou_thr=0.5,
                            min_pos_iou=0.5,
                        ),
                        sampler=dict(
                            type="RandomSampler", num=512, pos_fraction=0.25
                        ),
                        pos_weight=-1,
                    ),
                ),
                category_mapping=TEST_MAPPING,
            ),
            category_mapping=TEST_MAPPING,
        )
        self.assertTrue(isinstance(model, MMTwoStageDetector))
        inputs = [generate_input_sample(32, 32, 2, 3, use_score=False)]
        outs = model(inputs)
        self.assertTrue(isinstance(outs, dict))
        self.assertGreater(len(outs), 0)

    def test_one_stage_detector(self) -> None:
        """One stage detector test case."""
        model = MMOneStageDetector(
            backbone=MMDetBackbone(
                mm_cfg=dict(
                    type="ResNet",
                    depth=50,
                    num_stages=4,
                    out_indices=(0, 1, 2, 3),
                    norm_cfg=dict(type="BN", requires_grad=True),
                    style="pytorch",
                    init_cfg=dict(
                        type="Pretrained", checkpoint="torchvision://resnet50"
                    ),
                ),
                pixel_mean=PIXEL_MEAN,
                pixel_std=PIXEL_STD,
                neck=MMDetNeck(
                    mm_cfg=dict(
                        type="FPN",
                        in_channels=[256, 512, 1024, 2048],
                        out_channels=256,
                        num_outs=5,
                    ),
                ),
            ),
            bbox_head=MMDetDenseHead(
                mm_cfg=dict(
                    type="RetinaHead",
                    num_classes=80,
                    in_channels=256,
                    stacked_convs=4,
                    feat_channels=256,
                    anchor_generator=dict(
                        type="AnchorGenerator",
                        octave_base_scale=4,
                        scales_per_octave=3,
                        ratios=[0.5, 1.0, 2.0],
                        strides=[8, 16, 32, 64, 128],
                    ),
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[1.0, 1.0, 1.0, 1.0],
                    ),
                    loss_cls=dict(
                        type="FocalLoss",
                        use_sigmoid=True,
                        gamma=2.0,
                        alpha=0.25,
                    ),
                    loss_bbox=dict(type="L1Loss", loss_weight=1.0),
                    train_cfg=dict(
                        assigner=dict(
                            type="MaxIoUAssigner",
                            pos_iou_thr=0.5,
                            neg_iou_thr=0.4,
                            min_pos_iou=0,
                        ),
                    ),
                    test_cfg=dict(
                        nms_pre=1000,
                        min_bbox_size=0,
                        score_thr=0.05,
                        nms=dict(type="nms", iou_threshold=0.5),
                        max_per_img=100,
                    ),
                ),
                category_mapping=TEST_MAPPING,
            ),
            category_mapping=TEST_MAPPING,
        )
        self.assertTrue(isinstance(model, MMOneStageDetector))
