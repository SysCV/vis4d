"""Callback for switching the mode of YOLOX training."""
from __future__ import annotations

import os

import torch
from torch import nn

from vis4d.common import ArgsType
from vis4d.common.distributed import broadcast, get_rank
from vis4d.common.logging import rank_zero_info
from vis4d.engine.loss_module import LossModule
from vis4d.op.detect.yolox import YOLOXHead, YOLOXHeadLoss
from vis4d.op.loss.common import l1_loss

from .base import Callback
from .trainer_state import TrainerState


class YOLOXModeSwitchCallback(Callback):
    """Callback for switching the mode of YOLOX training."""

    def __init__(
        self, *args: ArgsType, switch_epoch: int, **kwargs: ArgsType
    ) -> None:
        """Init callback.

        Args:
            switch_epoch (int): Epoch to switch the mode.
        """
        super().__init__(*args, **kwargs)
        self.switch_epoch = switch_epoch
        self.switched = False

    def on_train_epoch_start(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
    ) -> None:
        """Hook to run at the beginning of a training epoch."""
        assert hasattr(model, "yolox_head") and isinstance(
            model.yolox_head, YOLOXHead
        ), "YOLOXModeSwitchCallback can only be used with YOLOX."
        found_loss = False
        for loss in loss_module.losses:
            if isinstance(loss["loss"], YOLOXHeadLoss):
                found_loss = True
                yolox_loss = loss["loss"]
                break
        assert found_loss, "YOLOXHeadLoss should be in LossModule."
        if (
            trainer_state["current_epoch"] + 1
        ) >= self.switch_epoch and not self.switched:
            rank_zero_info(
                "Switching YOLOX training mode (turning off strong "
                "augmentations, adding L1 loss)."
            )
            yolox_loss.loss_l1 = l1_loss
            self.switched = True
