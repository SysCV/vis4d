"""Test cases for Vis4D engine Trainer."""
import os
import shutil
from argparse import Namespace
from typing import Optional
from unittest import mock

import pytest
import torch
from pytorch_lightning.utilities.cli import SaveConfigCallback

from vis4d.data import BaseDataModule
from vis4d.data.dataset import ScalabelDataset
from vis4d.data.datasets import BDD100K
from vis4d.data.handler import BaseDatasetHandler
from vis4d.model import BaseModel
from vis4d.unittest.utils import get_test_file

# TODO update tests
from .trainer import BaseCLI, DefaultTrainer


class SampleDataModule(BaseDataModule):
    def __init__(self, task: str, *args, **kwargs):
        self.task = task
        super().__init__(*args, **kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
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
            f"bdd100k_{self.task}_sample", data_root, annotations, config_path
        )
        self.train_datasets = BaseDatasetHandler(
            ScalabelDataset(bdd100k_loader, True)
        )
        self.test_datasets = [
            BaseDatasetHandler(ScalabelDataset(bdd100k_loader, False))
        ]


class Model(BaseModel):
    def __init__(self, model_param: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_param = model_param


def _trainer_builder():
    return DefaultTrainer(
        work_dir="./unittests/", exp_name="test", fast_dev_run=True
    )


@pytest.mark.parametrize(
    ["trainer_class", "model_class"], [(_trainer_builder, Model)]
)
def test_base_cli(trainer_class, model_class, monkeypatch):
    """Test that CLI correctly instantiates model, trainer and calls fit."""
    expected_model = dict(model_param=7)
    expected_trainer = dict(limit_train_batches=100)

    def fit(trainer, model):
        for k, v in expected_model.items():
            assert getattr(model, k) == v
        for k, v in expected_trainer.items():
            assert getattr(trainer, k) == v
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

    with mock.patch("sys.argv", ["any.py", "fit", "--model.model_param=7"]):
        cli = BaseCLI(
            model_class,
            trainer_class=trainer_class,
            datamodule_class=SampleDataModule,
        )
        assert hasattr(cli.trainer, "ran_asserts") and cli.trainer.ran_asserts
