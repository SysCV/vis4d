"""Test cases for Vis4D engine."""
from __future__ import annotations

import shutil
import unittest

import pytest
import pytorch_lightning as pl
import torch
from _pytest.fixtures import FixtureRequest
from _pytest.monkeypatch import MonkeyPatch
from pytorch_lightning.utilities.cli import SaveConfigCallback
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from tests.util import MockModel
from vis4d.pl.data.base import DataModule
from vis4d.pl.optimizer import DefaultOptimizer
from vis4d.pl.trainer import CLI, DefaultTrainer


class MockDataset(Dataset):
    """Dataset mockup."""

    def __len__(self) -> int:
        """Len mockup."""
        return 10

    def __getitem__(self, index: int) -> Tensor:
        """Len mockup."""
        return torch.rand((3, 32, 32))


class MockDataModule(DataModule):
    """Data module Mockup."""

    def __init__(self, example: str, *args, **kwargs) -> None:
        """Creates an instance of the class."""
        super().__init__(*args, **kwargs)
        self.example = example

    def train_dataloader(self) -> DataLoader:
        """Mockup train dataloader."""
        dataset = MockDataset()
        return DataLoader(dataset, 1, True)

    def test_dataloader(self) -> list[DataLoader]:
        """Mockup test dataloader."""
        return self.train_dataloader()


def test_custom_init() -> None:
    """Test setup with some custom options like tqdm progress bar."""
    trainer = DefaultTrainer(
        work_dir="./unittests/",
        exp_name="trainer_test",
        callbacks=pl.callbacks.LearningRateMonitor(),
        tqdm=True,
        max_steps=2,
    )
    model = DefaultOptimizer(
        MockModel(model_param=7), MockModel(model_param=3)
    )
    trainer.fit(model, [None])


def test_cli(monkeypatch: MonkeyPatch) -> None:
    """Test that CLI correctly instantiates model/trainer and calls fit."""
    expected_model = dict(model_param=7)
    expected_trainer = dict(exp_name="cli_test")
    expected_datamodule = {"example": "attribute"}

    # wrap model into setup function to modify model_param via cmd line
    def model_setup(
        model_param: int = 7, optional_param: str | None = None
    ) -> DefaultOptimizer:
        return DefaultOptimizer(
            MockModel(model_param=model_param, optional_param=optional_param),
            MockModel(model_param=3),
        )

    def fit(trainer, model, datamodule):
        # do this because 'model' will be DefaultOptimizer, and we want to
        # check MockModel here
        model = model.model
        for k, v in expected_model.items():
            assert getattr(model, k) == v
        for k, v in expected_trainer.items():
            assert getattr(trainer, k) == v
        for k, v in expected_datamodule.items():
            assert getattr(datamodule, k) == v
        save_callback = [
            x for x in trainer.callbacks if isinstance(x, SaveConfigCallback)
        ]
        assert len(save_callback) == 1
        save_callback[0].on_train_start(trainer, model)

    def on_train_start(callback, trainer, _):
        config_dump = callback.parser.dump(callback.config, skip_none=False)
        for k, v in expected_model.items():
            assert f"  {k}: {v}" in config_dump
        for k, v in expected_trainer.items():
            assert f"  {k}: {v}" in config_dump
        trainer.ran_asserts = True

    monkeypatch.setattr(DefaultTrainer, "fit", fit)
    monkeypatch.setattr(SaveConfigCallback, "on_train_start", on_train_start)

    with unittest.mock.patch(
        "sys.argv",
        [
            "any.py",
            "fit",
            "--model.model_param=7",
            "--data.example=attribute",
            "--trainer.exp_name=cli_test",
            "--trainer.work_dir=./unittests/",
            "--trainer.max_steps=10",
            "--seed_everything=0",
        ],
    ):
        cli = CLI(model_setup, datamodule_class=MockDataModule)
        assert hasattr(cli.trainer, "ran_asserts") and cli.trainer.ran_asserts


@pytest.fixture(scope="module", autouse=True)
def teardown(request: FixtureRequest) -> None:
    """Clean up test files."""

    def remove_test_dir() -> None:
        shutil.rmtree("./unittests/", ignore_errors=True)

    request.addfinalizer(remove_test_dir)
