"""Test cases for progress bar."""
import unittest

from vis4d.common.progress import compose_log_str
from vis4d.common.time import Timer


class TestProgressBar(unittest.TestCase):
    """Test cases for progress bar functions."""

    def test_compose_log_str(self) -> None:
        """Test the compose_log_str function."""
        timer = Timer()
        num_batches = 10
        for i in range(1, num_batches + 1):
            prefix = "test"
            log_str = compose_log_str(
                prefix, i, num_batches, timer, {"accuracy": 1.0, "loss": 0.5}
            )
            self.assertTrue(log_str.startswith(prefix))
            log_list = log_str.split(", ")
            self.assertEqual(log_list[-2], "loss: 0.500")
            self.assertEqual(log_list[-1], "accuracy: 1.000")
