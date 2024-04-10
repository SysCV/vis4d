"""This module contains utilities for callbacks."""

from __future__ import annotations

import os

from torch import nn

from vis4d.common import ArgsType
from vis4d.common.distributed import broadcast
from vis4d.data.typing import DictData
from vis4d.engine.loss_module import LossModule
from vis4d.vis.base import Visualizer

from .base import Callback
from .trainer_state import TrainerState


class VisualizerCallback(Callback):
    """Callback for model visualization."""

    def __init__(
        self,
        *args: ArgsType,
        visualizer: Visualizer,
        visualize_train: bool = False,
        show: bool = False,
        save_to_disk: bool = True,
        save_prefix: str | None = None,
        **kwargs: ArgsType,
    ) -> None:
        """Init callback.

        Args:
            visualizer (Visualizer): Visualizer.
            visualize_train (bool): If the training data should be visualized.
                Defaults to False.
            save_prefix (str): Output directory for saving the visualizations.
            show (bool): If the visualizations should be shown. Defaults to
                False.
            save_to_disk (bool): If the visualizations should be saved to disk.
                Defaults to True.
        """
        super().__init__(*args, **kwargs)
        self.visualizer = visualizer
        self.visualize_train = visualize_train
        self.save_prefix = save_prefix
        self.show = show
        self.save_to_disk = save_to_disk

        if self.save_to_disk:
            assert (
                save_prefix is not None
            ), "If save_to_disk is True, save_prefix must be provided."
            self.output_dir = f"{self.save_prefix}/vis"

    def setup(self) -> None:  # pragma: no cover
        """Setup callback."""
        if self.save_to_disk:
            self.output_dir = broadcast(self.output_dir)

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
        cur_iter = batch_idx + 1

        if self.visualize_train:
            self.visualizer.process(
                cur_iter=cur_iter,
                **self.get_train_callback_inputs(outputs, batch),
            )

            if self.show:
                self.visualizer.show(cur_iter=cur_iter)

            if self.save_to_disk:
                os.makedirs(f"{self.output_dir}/train", exist_ok=True)
                self.visualizer.save_to_disk(
                    cur_iter=cur_iter,
                    output_folder=f"{self.output_dir}/train",
                )

            self.visualizer.reset()

    def on_test_batch_end(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        outputs: DictData,
        batch: DictData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Hook to run at the end of a testing batch."""
        cur_iter = batch_idx + 1

        self.visualizer.process(
            cur_iter=cur_iter,
            **self.get_test_callback_inputs(outputs, batch),
        )

        if self.show:
            self.visualizer.show(cur_iter=cur_iter)

        if self.save_to_disk:
            os.makedirs(f"{self.output_dir}/test", exist_ok=True)
            self.visualizer.save_to_disk(
                cur_iter=cur_iter,
                output_folder=f"{self.output_dir}/test",
            )

        self.visualizer.reset()
