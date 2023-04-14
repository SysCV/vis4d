# pylint: disable=consider-using-alias,consider-alternative-union-syntax
"""Data module composing the data loading pipeline."""
from __future__ import annotations

import lightning.pytorch as pl
from ml_collections import ConfigDict
from torch.utils import data

from vis4d.config.util import instantiate_classes
from vis4d.data.typing import DictData


class DataModule(pl.LightningDataModule):  # type: ignore
    """DataModule that wraps around the vis4d implementations.

    This is a wrapper around the vis4d implementations that allows to use
    pytorch-lightning for training and testing.
    """

    def __init__(
        self,
        data_cfg: ConfigDict,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.data_cfg = data_cfg

    def train_dataloader(self) -> data.DataLoader[DictData]:
        """Return dataloader for training."""
        return instantiate_classes(self.data_cfg.train_dataloader)

    def test_dataloader(self) -> list[data.DataLoader[DictData]]:
        """Return dataloaders for testing."""
        return instantiate_classes(self.data_cfg.test_dataloader)

    def val_dataloader(self) -> list[data.DataLoader[DictData]]:
        """Return dataloaders for validation."""
        return self.test_dataloader()
