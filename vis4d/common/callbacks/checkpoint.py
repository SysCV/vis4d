"""This module contains utilities for callbacks."""
from __future__ import annotations

import os

import torch
from torch import nn

from vis4d.common import ArgsType
from vis4d.common.distributed import broadcast, get_rank

from .base import Callback, CallbackInputs


class CheckpointCallback(Callback):
    """Callback for model checkpointing."""

    def __init__(
        self,
        *args: ArgsType,
        save_prefix: str,
        **kwargs: ArgsType,
    ) -> None:
        """Init callback.

        Args:
            save_prefix (str): Prefix of checkpoint path for saving.
            run_every_nth_epoch (int): Save model checkpoint every nth epoch.
                Defaults to 1.
            num_epochs (int): Number of total epochs, used for determining
                whether to visualize at the final epoch. Defaults to -1.
        """
        super().__init__(*args, **kwargs)
        self.output_dir = f"{save_prefix}/checkpoints"

    def setup(self) -> None:  # pragma: no cover
        """Setup callback."""
        self.output_dir = broadcast(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def on_train_epoch_end(
        self, callback_inputs: CallbackInputs, model: nn.Module
    ) -> None:
        """Hook to run at the end of a training epoch."""
        # TODO, save full state dict with optimizer, scheduler, etc.
        if get_rank() == 0:
            torch.save(
                model.state_dict(),
                f"{self.output_dir}/model_e{callback_inputs['epoch'] + 1}.pt",
            )
