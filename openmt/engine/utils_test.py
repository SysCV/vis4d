"""Test cases for detection engine config."""
import unittest
from argparse import Namespace

from openmt import config
from openmt.config import Dataset
from openmt.engine.utils import _register, to_detectron2
from openmt.unittest.utils import get_test_file


class TestConfig(unittest.TestCase):
    """Test cases for openmt detection config."""

    def test_register(self) -> None:
        """Testcase for register function."""
        datasets = [
            Dataset(
                **dict(
                    name="example",
                    type="MOTChallenge",
                    data_root="/path/to/data",
                    annotations="/path/to/annotations",
                )
            )
        ]
        names = _register(datasets)
        self.assertEqual(names, ["example"])

    def test_to_detectron2(self) -> None:
        """Testcase for detectron2 config conversion."""
        test_file = get_test_file("detect/faster_rcnn_R_50_FPN.toml")
        args = Namespace(config=test_file)
        cfg = config.parse_config(args)
        det2cfg = to_detectron2(cfg)
        self.assertEqual(
            det2cfg.SOLVER.IMS_PER_BATCH, cfg.solver.images_per_gpu
        )
        self.assertEqual(
            det2cfg.DATALOADER.NUM_WORKERS, cfg.dataloader.workers_per_gpu
        )
        self.assertEqual(det2cfg.DATASETS.TRAIN, ["bdd100k_det_sample_train"])
        self.assertEqual(det2cfg.DATASETS.TEST, ["bdd100k_det_sample_val"])
