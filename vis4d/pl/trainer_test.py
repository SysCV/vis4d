"""Test cases for Vis4D engine."""
import shutil
import unittest
from typing import Dict, List

import pytest
import pytorch_lightning as pl
from _pytest.fixtures import FixtureRequest
from _pytest.monkeypatch import MonkeyPatch
from pytorch_lightning.utilities.cli import SaveConfigCallback

from vis4d.data_to_revise.module_test import SampleDataModule
from vis4d.struct_to_revise import ArgsType

from ..model.optimize import DefaultOptimizer
from ..unittest.utils import MockModel
from .trainer import CLI, DefaultTrainer


def test_custom_init() -> None:
    """Test setup with some custom options like tqdm progress bar."""
    trainer = DefaultTrainer(
        work_dir="./unittests/",
        exp_name="trainer_test",
        callbacks=pl.callbacks.LearningRateMonitor(),
        tqdm=True,
        max_steps=2,
    )
    model = DefaultOptimizer(MockModel(model_param=7))
    trainer.fit(model, [None])


def test_base_cli(monkeypatch: MonkeyPatch) -> None:
    """Test that CLI correctly instantiates model/trainer and calls fit."""
    expected_model = dict(model_param=7)
    expected_trainer = dict(exp_name="cli_test")
    expected_datamodule = {"task": "track", "im_hw": (360, 640)}

    # wrap model into setup function to modify model_param via cmd line
    def model_setup(model_param: int = 7) -> DefaultOptimizer:
        return DefaultOptimizer(MockModel(model_param=model_param))

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
            "--trainer.exp_name=cli_test",
            "--trainer.work_dir=./unittests/",
            "--trainer.max_steps=10",
            "--data.task=track",
            "--data.im_hw=[360, 640]",
            "--seed_everything=0",
        ],
    ):
        cli = CLI(model_setup, datamodule_class=SampleDataModule)
        assert hasattr(cli.trainer, "ran_asserts") and cli.trainer.ran_asserts


@pytest.fixture(scope="module", autouse=True)
def teardown(request: FixtureRequest) -> None:
    """Clean up test files."""

    def remove_test_dir() -> None:
        shutil.rmtree("./unittests/", ignore_errors=True)

    request.addfinalizer(remove_test_dir)
