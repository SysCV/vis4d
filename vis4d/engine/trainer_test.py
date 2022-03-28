"""Test cases for Vis4D engine."""
import shutil
import unittest
from typing import List, Optional, Union

from _pytest.monkeypatch import MonkeyPatch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.cli import SaveConfigCallback

from vis4d.model import BaseModel
from vis4d.struct import ArgsType

from .trainer import BaseCLI, DefaultTrainer


class MockModel(BaseModel):
    """Model Mockup."""

    def __init__(self, model_param: int, *args: ArgsType, **kwargs: ArgsType):
        """Init."""
        super().__init__(*args, **kwargs)
        self.model_param = model_param


def _trainer_builder(
    exp_name: str,
    fast_dev_run: bool = False,
    callbacks: Optional[Union[List[Callback], Callback]] = None,
) -> DefaultTrainer:
    """Build mockup trainer."""
    return DefaultTrainer(
        work_dir="./unittests/",
        exp_name=exp_name,
        fast_dev_run=fast_dev_run,
        callbacks=callbacks,
    )


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
