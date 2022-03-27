"""Test cases for Vis4D models."""
import unittest
from typing import Optional

from vis4d.model.detect import (
    D2TwoStageDetector,
    MMOneStageDetector,
    MMTwoStageDetector,
)
from vis4d.model.track.graph import QDTrackGraph
from vis4d.model.track.similarity import QDSimilarityHead
from vis4d.unittest.utils import generate_input_sample

from .base import BaseModel
from .qdtrack import QDTrack


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
