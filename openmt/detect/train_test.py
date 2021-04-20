"""Test cases for detection engine training."""
from openmt import detect
from openmt.unittest.util import DetectTest, d2_data_reset


class TestTrain(DetectTest):
    """Test cases for openmt detection training."""

    def test_train(self) -> None:
        """Testcase for training."""
        self.assertIsNotNone(self.det2cfg)
        self.assertIsNotNone(self.cfg)
        detect.train_func(self.det2cfg, self.cfg)

    def test_train_launcher(self) -> None:
        """Testcase for training launcher."""
        d2_data_reset(self.det2cfg.DATASETS.TRAIN)
        d2_data_reset(self.det2cfg.DATASETS.TEST)
        detect.train(self.cfg)
