"""Test cases for time."""
import time
import unittest
from io import StringIO
from unittest.mock import patch

from vis4d.common.time import timeit


class TestTime(unittest.TestCase):
    """Test cases for time functions."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_timeit(self, stdout) -> None:
        """Test the timeit function."""

        @timeit
        def test_func() -> None:
            """Test function."""
            time.sleep(1.337)

        test_func()
        timed = stdout.getvalue()
        self.assertTrue(timed.startswith("test_func  "))
        self.assertTrue(timed.endswith(" ms\n"))
        self.assertAlmostEqual(float(timed[11:18]), 1337.0, delta=3.0)
