"""This module contains utilities for callbacks."""
from __future__ import annotations

import os

from torch import nn

from vis4d.common import DictStrAny
from vis4d.common.distributed import broadcast, get_rank
from vis4d.vis.base import Visualizer

from .base import Callback


class VisualizerCallback(Callback):
    """Callback for model visualization."""

    def __init__(
        self,
        visualizer: Visualizer,
        save_prefix: None | str = None,
        collect: str = "cpu",
        run_every_nth_epoch: int = 1,
        num_epochs: int = -1,
    ) -> None:
        """Init callback.

        Args:
            visualizer (Visualizer): Visualizer.
            save_prefix (str, Optional): Output directory for saving the
                visualizations. Defaults to None (no save).
            collect (str): Which device to collect results across GPUs on.
                Defaults to "cpu".
            run_every_nth_epoch (int): Visualize results every nth epoch.
                Defaults to 1.
            num_epochs (int): Number of total epochs, used for determining
                whether to visualize at the final epoch. Defaults to -1.
        """
        super().__init__(run_every_nth_epoch, num_epochs)
        assert collect in set(
            ("cpu", "gpu")
        ), f"Collect device {collect} unknown."
        self.collect = collect
        self.visualizer = visualizer
        self.save_prefix = save_prefix

        if self.save_prefix is not None:
            self.output_dir = f"{self.save_prefix}/vis"

    def setup(self) -> None:  # pragma: no cover
        """Setup callback."""
        if self.save_prefix is not None:
            self.output_dir = broadcast(self.output_dir)
            os.makedirs(self.output_dir, exist_ok=True)

    def on_test_epoch_end(
        self, model: nn.Module, epoch: None | int = None
    ) -> None:
        """Hook to run at the end of a testing epoch."""
        if get_rank() == 0:
            if self.save_prefix is not None:
                self.visualizer.save_to_disk(self.output_dir)
        self.visualizer.reset()

    def on_test_batch_end(
        self, model: nn.Module, shared_inputs: DictStrAny, inputs: DictStrAny
    ) -> None:
        """Hook to run at the end of a testing batch."""
        self.visualizer.process(**inputs)
