"""DataModule tests."""
import unittest
from typing import Optional

from vis4d.struct import ArgsType
from vis4d.unittest.utils import get_test_file

from .dataset import ScalabelDataset
from .datasets import BDD100K
from .handler import BaseDatasetHandler
from .module import BaseDataModule


class SampleDataModule(BaseDataModule):
    """Load sample data to test data pipelines."""

    def __init__(self, task: str, *args: ArgsType, **kwargs: ArgsType):
        """Init."""
        self.task = task
        super().__init__(*args, **kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data, setup data pipeline."""
        annotations = (
            f"vis4d/engine/testcases/{self.task}/bdd100k-samples/labels/"
        )
        data_root = (
            f"vis4d/engine/testcases/{self.task}/bdd100k-samples/images"
        )
        config_path = (
            f"vis4d/engine/testcases/{self.task}/bdd100k-samples/config.toml"
        )
        bdd100k_loader = BDD100K(
            f"bdd100k_{self.task}_sample",
            data_root,
            annotations,
            config_path=config_path,
        )
        self.train_datasets = BaseDatasetHandler(
            ScalabelDataset(bdd100k_loader, True)
        )
        self.test_datasets = [
            BaseDatasetHandler(ScalabelDataset(bdd100k_loader, False))
        ]


class TestDataModule(unittest.TestCase):
    """Test cases for base data module."""

    def test_track_data(self) -> None:
        """Test tracking data loading."""
        data_module = SampleDataModule("track")
        data_module.setup()
        for sample in data_module.train_dataloader():
            self.assertTrue(isinstance(sample, list))
            self.assertEqual(len(sample[0]), 1)

        # TODO continue
