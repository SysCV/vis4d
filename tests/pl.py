"""Pytorch lightning utilities for unit tests."""
from __future__ import annotations

from pytorch_lightning import Callback

from vis4d.pl import DefaultTrainer


def trainer_builder(
    exp_name: str,
    callbacks: None | list[Callback] | Callback = None,
) -> DefaultTrainer:
    """Build mockup trainer.

    Args:
        exp_name (str): Experiment name
        callbacks (list[Callback], Callback, optional): pl.callbacks that
                                                        should be executed
    """
    return DefaultTrainer(
        work_dir="./unittests/",
        exp_name=exp_name,
        callbacks=callbacks,
        max_steps=10,
    )
