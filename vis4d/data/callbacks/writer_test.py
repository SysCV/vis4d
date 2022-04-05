"""Writer callback tests."""
import unittest

from .writer import ScalabelWriterCallback


class TestScalabelWriterCallback(unittest.TestCase):
    """Test cases for ScalabelWriterCallback."""

    def test_write(self) -> None:
        """Test write."""
        writer = ScalabelWriterCallback(0, output_dir="./")

        # TODO continue
