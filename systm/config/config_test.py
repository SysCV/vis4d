"""Test cases for parsing config."""
import unittest

import yaml

from ..unittest.util import get_test_file
from .config import Detection


class TestLoadConfig(unittest.TestCase):
    """Test cases for BDD100K detection evaluation."""

    def test_det(self) -> None:
        """Check detection configuration."""
        config = yaml.load(
            open(get_test_file("config_det.yaml"), "r").read(),
            Loader=yaml.CLoader,
        )
        det_config = Detection(**config)
        self.assertEqual(det_config.model_name, "faster-rcnn")
        self.assertEqual(det_config.solver.base_lr, 0.02)
        self.assertEqual(det_config.solver.lr_policy, "step")
