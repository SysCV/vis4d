"""This module contains utilities for callbacks."""
from __future__ import annotations

import os

import torch
from torch import nn

from vis4d.common import ArgsType
from vis4d.common.distributed import broadcast, get_rank
from vis4d.engine.loss_module import LossModule

from .base import Callback
from .trainer_state import TrainerState


class CheckpointCallback(Callback):
    """Callback for model checkpointing."""

    def __init__(
        self, *args: ArgsType, save_prefix: str, **kwargs: ArgsType
    ) -> None:
        """Init callback.

        Args:
            save_prefix (str): Prefix of checkpoint path for saving.
        """
        super().__init__(*args, **kwargs)
        self.output_dir = f"{save_prefix}/checkpoints"

    def setup(self) -> None:  # pragma: no cover
        """Setup callback."""
        self.output_dir = broadcast(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def on_train_epoch_end(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
    ) -> None:
        """Hook to run at the end of a training epoch."""
        if get_rank() == 0:
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
