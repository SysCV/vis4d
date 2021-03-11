"""Test cases for detection engine module."""
import unittest

from systm import detect
from systm.unittest.util import EngineTest


class TestTrain(EngineTest):
    """Test cases for systm detection training."""

    def test_train(self) -> None:
        """Testcase for training."""
        if self.det2cfg is not None and self.cfg is not None:
            detect.train_func(self.det2cfg, self.cfg)
        else:
            self.assertEqual(True, False, msg="failed to initialize configs!")


if __name__ == "__main__":
    unittest.main()
