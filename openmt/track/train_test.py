"""Test cases for tracking engine training."""

from detectron2.data import DatasetCatalog

from openmt.unittest.util import TrackTest

from .train import track_train_func, train


class TestTrain(TrackTest):
    """Test cases for openmt tracking training."""

    def test_train(self) -> None:
        """Testcase for training."""
        self.assertIsNotNone(self.det2cfg)
        self.assertIsNotNone(self.cfg)
        track_train_func(self.det2cfg, self.cfg)

    def test_train_launcher(self) -> None:
        """Testcase for training launcher."""
        for ds in self.det2cfg.DATASETS.TRAIN:
            DatasetCatalog.remove(ds)
        for ds in self.det2cfg.DATASETS.TEST:
            DatasetCatalog.remove(ds)
        train(self.cfg)
