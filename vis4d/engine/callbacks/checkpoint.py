"""This module contains utilities for callbacks."""

from __future__ import annotations

import os

import torch
from torch import nn

from vis4d.common import ArgsType
from vis4d.common.distributed import broadcast, rank_zero_only
from vis4d.data.typing import DictData
from vis4d.engine.callbacks.trainer_state import TrainerState
from vis4d.engine.loss_module import LossModule

from .base import Callback
from .trainer_state import TrainerState


class CheckpointCallback(Callback):
    """Callback for model checkpointing."""

    def __init__(
        self,
        *args: ArgsType,
        save_prefix: str,
        checkpoint_period: int = 1,
        **kwargs: ArgsType,
    ) -> None:
        """Init callback.

        Args:
            save_prefix (str): Prefix of checkpoint path for saving.
            checkpoint_period (int, optional): Checkpoint period. Defaults to
                1.
        """
        super().__init__(*args, **kwargs)
        self.output_dir = f"{save_prefix}/checkpoints"
        self.checkpoint_period = checkpoint_period

    def setup(self) -> None:  # pragma: no cover
        """Setup callback."""
        self.output_dir = broadcast(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    @rank_zero_only
    def _save_checkpoint(
        self, trainer_state: TrainerState, model: nn.Module
    ) -> None:
        """Save checkpoint."""
        epoch = trainer_state["current_epoch"]
        step = trainer_state["global_step"]
        ckpt_dict = {
            "epoch": epoch,
            "global_step": step,
            "state_dict": model.state_dict(),
        }

        if "optimizers" in trainer_state:
            ckpt_dict["optimizers"] = [
                optimizer.state_dict()
                for optimizer in trainer_state["optimizers"]
            ]

        if "lr_schedulers" in trainer_state:
            ckpt_dict["lr_schedulers"] = [
                lr_scheduler.state_dict()
                for lr_scheduler in trainer_state["lr_schedulers"]
            ]

        torch.save(
            ckpt_dict,
            f"{self.output_dir}/epoch={epoch}-step={step}.ckpt",
        )

        torch.save(ckpt_dict, f"{self.output_dir}/last.ckpt")

    def on_train_batch_end(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
        outputs: DictData,
        batch: DictData,
        batch_idx: int,
    ) -> None:
        """Hook to run at the end of a training batch."""
        if (
            not self.epoch_based
            and trainer_state["global_step"] % self.checkpoint_period == 0
        ):
            self._save_checkpoint(trainer_state, model)

    def on_train_epoch_end(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
    ) -> None:
        """Hook to run at the end of a training epoch."""
        if (
            self.epoch_based
            and (trainer_state["current_epoch"] + 1) % self.checkpoint_period
            == 0
        ):
            self._save_checkpoint(trainer_state, model)
