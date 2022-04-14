"""Testcases for base dataset."""
import os
import unittest

from scalabel.label.utils import compare_results

from .scalabel import Scalabel


class TestBaseDatasetLoader(unittest.TestCase):
    """build Testcase class."""

    data_root = "vis4d/engine/testcases/track/bdd100k-samples/"

    def test_load_cached(self) -> None:
        """Test load_cached_dataset function."""
        dataset_loader1 = Scalabel(
            "test_dataset",
            f"{self.data_root}/images",
            f"{self.data_root}/labels",
            config_path=f"{self.data_root}/config.toml",
            cache_as_binary=True,
        )

        cache_path = f"{self.data_root}/labels".rstrip("/") + ".pkl"
        self.assertTrue(os.path.exists(cache_path))

        dataset_loader2 = Scalabel(
            "test_dataset",
            f"{self.data_root}/images",
            f"{self.data_root}/labels",
            config_path=f"{self.data_root}/config.toml",
            cache_as_binary=True,
        )
        compare_results(dataset_loader1.frames, dataset_loader2.frames)
        os.remove(cache_path)
