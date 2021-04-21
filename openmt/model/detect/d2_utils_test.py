"""Testcases for d2 utils."""
import unittest

from openmt.unittest.util import get_test_file

from .d2_utils import D2GeneralizedRCNNConfig, model_to_detectron2


class TestD2Utils(unittest.TestCase):
    """Testcases for d2 utils."""

    def test_model_tod2(self) -> None:
        """Testcase for config to d2 config."""
        cfg = D2GeneralizedRCNNConfig(
            type="D2GeneralizedRCNN",
            model_base="faster-rcnn/r50-fpn",
            num_classes=10,
        )
        model_to_detectron2(cfg)
        cfg.model_base = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        cfg.override_mapping = True
        det2cfg = model_to_detectron2(cfg)
        self.assertEqual(det2cfg.MODEL.META_ARCHITECTURE, "GeneralizedRCNN")
        cfg.override_mapping = False

        cfg.model_base = get_test_file("test-cfg.yaml")
        det2cfg = model_to_detectron2(cfg)
        self.assertEqual(det2cfg.MODEL.META_ARCHITECTURE, "GeneralizedRCNN")
