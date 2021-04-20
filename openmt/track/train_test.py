"""Test cases for tracking engine training."""

from openmt.unittest.util import TrackTest, d2_data_reset

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
        d2_data_reset(self.det2cfg.DATASETS.TRAIN)
        d2_data_reset(self.det2cfg.DATASETS.TEST)
        train(self.cfg)
