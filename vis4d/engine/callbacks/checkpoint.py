"""This module contains utilities for callbacks."""
from __future__ import annotations

import os

import torch
from torch import nn

from vis4d.common import ArgsType
from vis4d.common.distributed import broadcast, get_rank

from .base import Callback
from .trainer_state import TrainerState


class CheckpointCallback(Callback):
    """Callback for model checkpointing."""

    def __init__(
        self,
        *args: ArgsType,
        save_prefix: str,
        save_ckpt_every_n_epoch: int = 1,
        **kwargs: ArgsType,
    ) -> None:
        """Init callback.

        Args:
            save_prefix (str): Prefix of checkpoint path for saving.
            save_ckpt_every_n_epoch (int, optional): Save checkpoint every
                n epochs. Defaults to 1.
        """
        super().__init__(*args, **kwargs)
        self.output_dir = f"{save_prefix}/checkpoints"
        self.save_ckpt_every_n_epoch = save_ckpt_every_n_epoch

    def setup(self) -> None:  # pragma: no cover
        """Setup callback."""
        self.output_dir = broadcast(self.output_dir)

    def _run_on_epoch(self, current_epoch: int) -> bool:
        """Return whether save checkpoint on current epoch.

        Args:
            current_epoch (int): Current epoch.
        """
        return current_epoch % self.save_ckpt_every_n_epoch == 0

    def on_train_epoch_end(
        self, trainer_state: TrainerState, model: nn.Module
    ) -> None:
        """Hook to run at the end of a training epoch."""
        # TODO, save full state dict with optimizer, scheduler, etc.
        if get_rank() != 0:
            return
        current_epoch = trainer_state["current_epoch"] + 1
        if self._run_on_epoch(current_epoch):
            os.makedirs(self.output_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                f"{self.output_dir}/model_e{current_epoch}.pt",
            )
