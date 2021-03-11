"""Test cases for detection engine training."""

from detectron2.data import DatasetCatalog

from openmt import detect
from openmt.unittest.util import DetectTest


class TestTrain(DetectTest):
    """Test cases for openmt detection training."""

    def test_train(self) -> None:
        """Testcase for training."""
        self.assertIsNotNone(self.det2cfg)
        self.assertIsNotNone(self.cfg)
        detect.train_func(self.det2cfg, self.cfg)

    def test_train_launcher(self) -> None:
        """Testcase for training launcher."""
        for ds in self.det2cfg.DATASETS.TRAIN:
            DatasetCatalog.remove(ds)
        for ds in self.det2cfg.DATASETS.TEST:
            DatasetCatalog.remove(ds)
        detect.train(self.cfg)
