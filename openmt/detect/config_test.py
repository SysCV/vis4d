"""Test cases for detection engine config."""
import unittest
from argparse import Namespace

from openmt import config
from openmt.config import Dataset, DatasetType
from openmt.detect.config import _register, to_detectron2
from openmt.unittest.util import get_test_file


class TestConfig(unittest.TestCase):
    """Test cases for openmt detection config."""

    def test_register(self) -> None:
        """Testcase for register function."""
        datasets = [
            Dataset(
                **dict(
                    name="example",
                    type=DatasetType.CUSTOM,
                    data_root="/path/to/data",
                )
            )
        ]
        self.assertRaises(NotImplementedError, _register, datasets)

    def test_to_detectron2(self) -> None:
        """Testcase for detectron2 config conversion."""
        test_file = get_test_file("retinanet_R_50_FPN.toml")
        args = Namespace(config=test_file)
        cfg = config.parse_config(args)

        cfg.detection.model_base = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
        cfg.detection.override_mapping = True
        det2cfg = to_detectron2(cfg)
        self.assertEqual(det2cfg.MODEL.META_ARCHITECTURE, "RetinaNet")
        cfg.detection.override_mapping = False

        cfg.detection.model_base = get_test_file("retinanet_R_50_FPN_3x.yaml")
        cfg.detection.weights = None
        det2cfg = to_detectron2(cfg)
        self.assertEqual(det2cfg.MODEL.META_ARCHITECTURE, "RetinaNet")

        cfg.detection.weights = test_file  # set to any existing file
        det2cfg = to_detectron2(cfg)
        self.assertEqual(det2cfg.MODEL.WEIGHTS, test_file)

        cfg.detection.weights = ""  # set to non-existing file
        self.assertRaises(ValueError, to_detectron2, cfg)
