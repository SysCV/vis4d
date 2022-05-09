"""Test cases for Vis4D engine."""
import shutil
import unittest
from typing import Dict, List

import pytest
import pytorch_lightning as pl
from _pytest.fixtures import FixtureRequest
from _pytest.monkeypatch import MonkeyPatch
from pytorch_lightning.utilities.cli import SaveConfigCallback

from vis4d.data.module_test import SampleDataModule
from vis4d.struct import ArgsType

from ..unittest.utils import MockModel
from .trainer import BaseCLI, DefaultTrainer


def test_custom_init() -> None:
    """Test setup with some custom options like tqdm progress bar."""
    trainer = DefaultTrainer(
        work_dir="./unittests/",
        exp_name="trainer_test",
        callbacks=pl.callbacks.LearningRateMonitor(),
        tqdm=True,
        max_steps=2,
    )
    model = MockModel(model_param=7)
    trainer.fit(model, [None])


def test_tune(monkeypatch: MonkeyPatch) -> None:
    """Test tune function."""
    model_params = [0, 1, 2, 3]
    trainer = DefaultTrainer(
        work_dir="./unittests/",
        exp_name="test_tune",
        tuner_params={"model_param": model_params},
        tuner_metrics=["my_metric"],
    )
    model = MockModel(model_param=7)

    model_param_vals = []

    def dummy_test(  # pylint: disable=unused-argument
        *args: ArgsType, **kwargs: ArgsType
    ) -> List[Dict[str, float]]:
        model_param_vals.append(model.model_param)
        return [{"my_metric": 0}]

    monkeypatch.setattr(DefaultTrainer, "test", dummy_test)
    trainer.tune(model)
    assert set(model_param_vals) == set(model_params)


def test_base_cli(monkeypatch: MonkeyPatch) -> None:
    """Test that CLI correctly instantiates model/trainer and calls fit."""
    expected_model = dict(model_param=7)
    expected_trainer = dict(exp_name="cli_test")
    expected_datamodule = {"task": "track", "im_hw": (360, 640)}

    def fit(trainer, model, datamodule):
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
        cli = BaseCLI(MockModel, datamodule_class=SampleDataModule)
        assert hasattr(cli.trainer, "ran_asserts") and cli.trainer.ran_asserts


@pytest.fixture(scope="module", autouse=True)
def teardown(request: FixtureRequest) -> None:
    """Clean up test files."""

    def remove_test_dir() -> None:
        shutil.rmtree("./unittests/", ignore_errors=True)

    request.addfinalizer(remove_test_dir)
