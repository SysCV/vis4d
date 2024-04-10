"""Callback for updating EMA model."""

from __future__ import annotations

from torch import nn

from vis4d.common.distributed import is_module_wrapper
from vis4d.common.typing import MetricLogs
from vis4d.data.typing import DictData
from vis4d.engine.loss_module import LossModule
from vis4d.model.adapter import ModelEMAAdapter

from .base import Callback
from .trainer_state import TrainerState


class EMACallback(Callback):
    """Callback for EMA."""

    def on_train_batch_end(  # pylint: disable=useless-return
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
        outputs: DictData,
        batch: DictData,
        batch_idx: int,
    ) -> None | MetricLogs:
        """Hook to run at the end of a training batch."""
        if is_module_wrapper(model):
            module = model.module
        else:
            module = model
        assert isinstance(module, ModelEMAAdapter), (
            "Model should be wrapped with ModelEMAAdapter when using "
            "EMACallback."
        )
        module.update(trainer_state["global_step"])
        return None
