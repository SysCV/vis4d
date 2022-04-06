"""Writer callback tests."""
import unittest

from ..datasets import Scalabel
from .writer import ScalabelWriterCallback


class TestScalabelWriterCallback(unittest.TestCase):
    """Test cases for ScalabelWriterCallback."""

    def test_write(self) -> None:
        """Test write."""
        base_dir = "vis4d/engine/testcases/track/bdd100k-samples"
        dataset_loader = Scalabel(
            "test_dataset",
            f"{base_dir}/images",
            f"{base_dir}/labels",
            config_path=f"{base_dir}/config.toml",
            eval_metrics=["detect"],
        )
        writer = ScalabelWriterCallback(0, output_dir="./")
