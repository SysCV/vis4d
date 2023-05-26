"""Data module composing the data loading pipeline."""
from __future__ import annotations

import lightning.pytorch as pl
from ml_collections import ConfigDict
from torch.utils.data import DataLoader

from vis4d.config import instantiate_classes
from vis4d.data.typing import DictData


class DataModule(pl.LightningDataModule):  # type: ignore
    """DataModule that wraps around the vis4d implementations.

    This is a wrapper around the vis4d implementations that allows to use
    pytorch-lightning for training and testing.
    """

    def __init__(self, data_cfg: ConfigDict) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.data_cfg = data_cfg

    def train_dataloader(self) -> DataLoader[DictData]:
        """Return dataloader for training."""
        return instantiate_classes(self.data_cfg.train_dataloader)

    def test_dataloader(self) -> list[DataLoader[DictData]]:
        """Return dataloaders for testing."""
        return instantiate_classes(self.data_cfg.test_dataloader)

    def val_dataloader(self) -> list[DataLoader[DictData]]:
        """Return dataloaders for validation."""
        return self.test_dataloader()
