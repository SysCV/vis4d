"""Callback for updating EMA model."""

from __future__ import annotations

import lightning.pytorch as pl

from vis4d.common.distributed import is_module_wrapper
from vis4d.data.typing import DictData
from vis4d.model.adapter import ModelEMAAdapter

from .base import Callback
from .util import get_model


class EMACallback(Callback):
    """Callback for EMA."""

    def on_train_batch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: DictData,
        batch: DictData,
        batch_idx: int,
    ) -> None:
        """Hook to run at the end of a training batch."""
        model = get_model(pl_module)

        if is_module_wrapper(model):
            module = model.module
        else:
            module = model

        assert isinstance(module, ModelEMAAdapter), (
            "Model should be wrapped with ModelEMAAdapter when using "
            "EMACallback."
        )

        module.update(trainer.global_step)
