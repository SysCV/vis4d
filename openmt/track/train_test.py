"""Test cases for tracking engine training."""

from openmt.unittest.util import TrackTest

from .train import train


class TestTrain(TrackTest):
    """Test cases for openmt tracking training."""

    def test_train(self) -> None:
        """Testcase for training."""
        self.assertIsNotNone(self.cfg)
        self.cfg.launch.action = "train"
        train(self.cfg)
