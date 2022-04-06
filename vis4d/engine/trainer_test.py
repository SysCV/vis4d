"""Test cases for Vis4D engine."""
import shutil
import unittest
from typing import Dict, List

import pytorch_lightning as pl
from _pytest.monkeypatch import MonkeyPatch
from pytorch_lightning.utilities.cli import SaveConfigCallback

from vis4d.struct import ArgsType

from ..unittest.utils import MockModel, _trainer_builder
from .trainer import BaseCLI, DefaultTrainer


class TestTrainer(unittest.TestCase):
    """Testcases for DefaultTrainer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.monkeypatch = MonkeyPatch()

    def test_custom_init(self) -> None:
        """Test setup with some custom options like tqdm progress bar."""
        DefaultTrainer(
            work_dir="./unittests/",
            exp_name="trainer_test",
            fast_dev_run=True,
            callbacks=pl.callbacks.LearningRateMonitor(),
        )

    def test_tune(self) -> None:
        """Test tune function."""
        trainer = DefaultTrainer(
            work_dir="./unittests/",
            exp_name="test_tune",
            tuner_params={"model_param": [0, 1, 2, 3]},
            tuner_metrics=["my_metric"],
        )
        model = MockModel(model_param=7)

        def dummy_test(  # pylint: disable=unused-argument
            *args: ArgsType, **kwargs: ArgsType
        ) -> List[Dict[str, float]]:
            return [{"my_metric": model.model_param}]

        self.monkeypatch.setattr(DefaultTrainer, "test", dummy_test)
        trainer.tune(model)


class TestCLI(unittest.TestCase):
    """Test cases for vis4d cli."""

    def setUp(self) -> None:
        """Set up test case."""
        self.monkeypatch = MonkeyPatch()

    def test_base_cli(self) -> None:
        """Test that CLI correctly instantiates model/trainer and calls fit."""
        expected_model = dict(model_param=7)
        expected_trainer = dict(exp_name="cli_test")

        def fit(trainer, model):  # type: ignore
            for k, v in expected_model.items():
                assert getattr(model, k) == v
            for k, v in expected_trainer.items():  # type: ignore # pylint: disable=line-too-long
                assert getattr(trainer, k) == v
            save_callback = [
                x
                for x in trainer.callbacks
                if isinstance(x, SaveConfigCallback)
            ]
            assert len(save_callback) == 1
            save_callback[0].on_train_start(trainer, model)

        def on_train_start(callback, trainer, _):  # type: ignore
            config_dump = callback.parser.dump(
                callback.config, skip_none=False
            )
            for k, v in expected_model.items():
                assert f"  {k}: {v}" in config_dump
            for k, v in expected_trainer.items():  # type: ignore
                assert f"  {k}: {v}" in config_dump
            trainer.ran_asserts = True

        self.monkeypatch.setattr(DefaultTrainer, "fit", fit)
        self.monkeypatch.setattr(
            SaveConfigCallback, "on_train_start", on_train_start
        )

        with unittest.mock.patch(
            "sys.argv",
            [
                "any.py",
                "fit",
                "--model.model_param=7",
                "--trainer.exp_name=cli_test",
                "--seed_everything=0",
            ],
        ):
            cli = BaseCLI(MockModel, trainer_class=_trainer_builder)
            assert (
                hasattr(cli.trainer, "ran_asserts") and cli.trainer.ran_asserts  # type: ignore # pylint: disable=line-too-long
            )

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up test files."""
        shutil.rmtree("./unittests/", ignore_errors=True)
