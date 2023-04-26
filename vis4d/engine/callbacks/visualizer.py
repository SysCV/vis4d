"""This module contains utilities for callbacks."""
from __future__ import annotations

import os

from torch import nn

from vis4d.common import ArgsType
from vis4d.common.distributed import broadcast, get_rank
from vis4d.data.typing import DictData
from vis4d.engine.connectors.util import get_inputs_for_pred_and_data
from vis4d.vis.base import Visualizer

from .base import Callback
from .trainer_state import TrainerState


# TODO: Refactor this to save per batch
class VisualizerCallback(Callback):
    """Callback for model visualization."""

    def __init__(
        self,
        *args: ArgsType,
        visualizer: Visualizer,
        save_prefix: None | str = None,
        **kwargs: ArgsType,
    ) -> None:
        """Init callback.

        Args:
            visualizer (Visualizer): Visualizer.
            save_prefix (str | None, optional): Output directory for saving the
                visualizations. Defaults to None (no save).
        """
        super().__init__(*args, **kwargs)
        self.visualizer = visualizer
        self.save_prefix = save_prefix

        if self.save_prefix is not None:
            self.output_dir = f"{self.save_prefix}/vis"

    def setup(self) -> None:  # pragma: no cover
        """Setup callback."""
        if self.save_prefix is not None:
            self.output_dir = broadcast(self.output_dir)

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
        self.visualizer.process(
            **get_inputs_for_pred_and_data(
                self.connector,
                outputs,
                batch,
            )
        )

    def on_test_epoch_end(
        self, trainer_state: TrainerState, model: nn.Module
    ) -> None:
        """Hook to run at the end of a testing epoch."""
        if get_rank() == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            if self.save_prefix is not None:
                self.visualizer.save_to_disk(self.output_dir)
        self.visualizer.reset()
