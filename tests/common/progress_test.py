"""Test cases for progress bar."""
import unittest

import torch

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
            metrics = {
                "accuracy": 0.5,
                "loss": torch.tensor(0.00001),
                "num": 5,
            }
            log_str = compose_log_str(prefix, i, num_batches, timer, metrics)
            self.assertTrue(log_str.startswith(prefix))
            log_list = log_str.split(", ")
            self.assertEqual(log_list[-3], "loss: 1.000e-05")
            self.assertEqual(log_list[-2], "accuracy: 0.5000")
            self.assertEqual(log_list[-1], "num: 5")
