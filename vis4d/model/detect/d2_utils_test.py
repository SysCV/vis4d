"""Testcases for d2 utils."""
import unittest

from vis4d.unittest.utils import get_test_file

from .d2_utils import model_to_detectron2


class TestD2Utils(unittest.TestCase):
    """Testcases for d2 utils."""

    def test_model_tod2(self) -> None:
        """Testcase for config to d2 config."""
        model_to_detectron2(
            "faster-rcnn/r50-fpn", category_mapping={"car": 0, "pedestrian": 1}
        )
        det2cfg = model_to_detectron2(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
            override_mapping=True,
            category_mapping={"car": 0, "pedestrian": 1},
        )
        self.assertEqual(det2cfg.MODEL.META_ARCHITECTURE, "GeneralizedRCNN")

        det2cfg = model_to_detectron2(
            get_test_file("test-cfg.yaml"),
            category_mapping={"car": 0, "pedestrian": 1},
        )
        self.assertEqual(det2cfg.MODEL.META_ARCHITECTURE, "GeneralizedRCNN")
