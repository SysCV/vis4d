"""Utilities for unit tests."""
from __future__ import annotations

from pytorch_lightning import Callback

from vis4d.pl import DefaultTrainer


def _trainer_builder(
    exp_name: str,
    fast_dev_run: bool = False,
    callbacks: None | list[Callback] | Callback = None,
) -> DefaultTrainer:
    """Build mockup trainer."""
    return DefaultTrainer(
        work_dir="./unittests/",
        exp_name=exp_name,
        fast_dev_run=fast_dev_run,
        callbacks=callbacks,
        max_steps=10,
    )
