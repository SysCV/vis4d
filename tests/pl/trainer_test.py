"""Pytorch lightning utilities for unit tests."""
from __future__ import annotations

import unittest

from pytorch_lightning import Callback

from torch import nn, optim

from vis4d.config.optimizer import get_optimizer_config
from vis4d.config.util import ConfigDict, class_config

from vis4d.pl import DefaultTrainer
from vis4d.pl.training_module import TrainingModule

from ..util import MockModel


def get_trainer(
    exp_name: str, callbacks: None | list[Callback] = None
) -> DefaultTrainer:
    """Build mockup trainer.

    Args:
        exp_name (str): Experiment name
        callbacks (list[Callback], Callback, optional): pl.callbacks that
                                                        should be executed
    """
    if callbacks is None:
        callbacks = []

    return DefaultTrainer(
        work_dir="./unittests/",
        exp_name=exp_name,
        version="test",
        callbacks=callbacks,
        max_steps=2,
        devices=0,
    )


def get_training_module(model: nn.Module):
    """Build mockup training module.

    Args:
        model (nn.Module): Pytorch model
    """
    optimizer_cfg = get_optimizer_config(class_config(optim.SGD, lr=0.01))
    return TrainingModule(model, [optimizer_cfg], None, None, seed=1)


class PLTrainerTest(unittest.TestCase):
    """Pytorch lightning trainer test class."""

    trainer = get_trainer("test")
    training_module = get_training_module(MockModel(model_param=7))
    trainer.fit(training_module, [None])
