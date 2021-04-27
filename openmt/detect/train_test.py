"""Test cases for detection engine training."""
from openmt import detect
from openmt.unittest.util import DetectTest


class TestTrain(DetectTest):
    """Test cases for openmt detection training."""

    def test_train(self) -> None:
        """Testcase for training."""
        self.assertIsNotNone(self.cfg)
        self.cfg.launch.action = "train"
        detect.train(self.cfg)
