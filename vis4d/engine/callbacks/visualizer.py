"""This module contains utilities for callbacks."""
from __future__ import annotations

import os

from torch import nn

from vis4d.common import ArgsType
from vis4d.common.distributed import broadcast
from vis4d.data.typing import DictData
from vis4d.vis.base import Visualizer

from .base import Callback
from .trainer_state import TrainerState


class VisualizerCallback(Callback):
    """Callback for model visualization."""

    def __init__(
        self,
        *args: ArgsType,
        visualizer: Visualizer,
        show: bool = False,
        save_to_disk: bool = True,
        save_prefix: str | None = None,
        vis_freq: int = 1,
        **kwargs: ArgsType,
    ) -> None:
        """Init callback.

        Args:
            visualizer (Visualizer): Visualizer.
            save_prefix (str): Output directory for saving the visualizations.
            show (bool): If the visualizations should be shown. Defaults to
                False.
            save_to_disk (bool): If the visualizations should be saved to disk.
                Defaults to True.
            vis_freq (int): Frequency of visualizations. Defaults to 1.
        """
        super().__init__(*args, **kwargs)
        self.visualizer = visualizer
        self.save_prefix = save_prefix
        self.show = show
        self.save_to_disk = save_to_disk
        self.vis_freq = vis_freq

        if self.save_to_disk:
            assert (
                save_prefix is not None
            ), "If save_to_disk is True, save_prefix must be provided."
            self.output_dir = f"{self.save_prefix}/vis"

    def setup(self) -> None:  # pragma: no cover
        """Setup callback."""
        if self.save_to_disk:
            self.output_dir = broadcast(self.output_dir)
            os.makedirs(self.output_dir, exist_ok=True)

    def on_train_epoch_start(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
    ) -> None:
        """Hook to run at the start of a training epoch."""
        self.visualizer.reset()

    def on_train_batch_end(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        outputs: DictData,
        batch: DictData,
        batch_idx: int,
    ) -> None:
        """Hook to run at the end of a training batch."""
        if self.train_connector is not None:
            cur_iter = batch_idx + 1

            if cur_iter % self.vis_freq == 0:
                self.visualizer.process(
                    **self.get_data_connector_results(
                        outputs,
                        batch,
                        train=True,
                    )
                )

                if self.show:
                    self.visualizer.show()

                if self.save_to_disk:
                    train_folder = f"{self.output_dir}/train"
                    os.makedirs(train_folder, exist_ok=True)
                    self.visualizer.save_to_disk(train_folder)

                self.visualizer.reset()

    def on_test_epoch_start(
        self, trainer_state: TrainerState, model: nn.Module
    ) -> None:
        """Hook to run at the start of a testing epoch."""
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

        if cur_iter % self.vis_freq == 0:
            self.visualizer.process(
                **self.get_data_connector_results(
                    outputs,
                    batch,
                    train=False,
                )
            )

            if self.show:
                self.visualizer.show()

            if self.save_to_disk:
                test_folder = f"{self.output_dir}/test"
                os.makedirs(test_folder, exist_ok=True)
                self.visualizer.save_to_disk(test_folder)

            self.visualizer.reset()
