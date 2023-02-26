"""This module contains utilities for callbacks."""
from __future__ import annotations

import os
from collections import defaultdict

import torch
from torch import nn

from vis4d.common import DictStrAny, MetricLogs
from vis4d.common.distributed import all_gather_object_cpu, get_rank
from vis4d.common.logging import rank_zero_info
from vis4d.common.progress import compose_log_str
from vis4d.common.time import Timer
from vis4d.eval.base import Evaluator
from vis4d.vis.base import Visualizer


class Callback:
    """Base class for Vis4D Callbacks."""

    def __init__(
        self, run_every_nth_epoch: int = 1, num_epochs: int = -1
    ) -> None:
        """Init callback.

        Args:
            run_every_nth_epoch (int): Evaluate model every nth epoch.
                Defaults to 1.
            num_epochs (int): Number of total epochs, used for determining
                whether to evaluate at the final epoch. Defaults to -1.
        """
        self.run_every_nth_epoch = run_every_nth_epoch
        self.num_epochs = num_epochs

    def run_on_epoch(self, epoch: int | None) -> bool:
        """Returns whether to run callback for current epoch (default True)."""
        return (
            epoch is None
            or epoch == self.num_epochs - 1
            or epoch % self.run_every_nth_epoch == self.run_every_nth_epoch - 1
        )

    def on_train_epoch_start(self, model: nn.Module, epoch: int) -> None:
        """Hook to run at the beginning of a training epoch.

        Args:
            model (nn.Module): Model that is being trained.
            epoch (int): Current training epoch.
        """

    def on_train_epoch_end(self, model: nn.Module, epoch: int) -> None:
        """Hook to run at the end of a training epoch.

        Args:
            model (nn.Module): Model that is being trained.
            epoch (int): Current training epoch.
        """

    def on_train_batch_end(
        self,
        model: nn.Module,
        inputs: DictStrAny,
        metrics: DictStrAny,
        cur_epoch: int,
        total_epochs: int,
        cur_batch: int,
        total_batches: int,
    ) -> None:
        """Hook to run at the end of a training batch."""

    def on_test_epoch_start(self, model: nn.Module, epoch: int) -> None:
        """Hook to run at the beginning of a testing epoch.

        Args:
            model (nn.Module): Model that is being trained.
            epoch (int): Current training epoch.
        """

    def on_test_epoch_end(self, model: nn.Module, epoch: int) -> None:
        """Hook to run at the end of a testing epoch."""

    def on_test_batch_end(self, model: nn.Module, inputs: DictStrAny) -> None:
        """Hook to run at the end of a testing batch.

        Args:
            model: Model that is being trained.
            inputs (ArgsType): Inputs for callback.
        """


class EvaluatorCallback(Callback):
    """Callback for model evaluation."""

    def __init__(
        self,
        evaluator: Evaluator,
        output_dir: None | str = None,
        collect: str = "cpu",
        run_every_nth_epoch: int = 1,
        num_epochs: int = -1,
    ) -> None:
        """Init callback.

        Args:
            evaluator (Evaluator): Evaluator.
            output_dir (str, Optional): Output directory for saving the
                evaluation results. Defaults to None (no save).
            collect (str): Which device to collect results across GPUs on.
                Defaults to "cpu".
            run_every_nth_epoch (int): Evaluate model every nth epoch.
                Defaults to 1.
            num_epochs (int): Number of total epochs, used for determining
                whether to evaluate at the final epoch. Defaults to -1.
        """
        super().__init__(run_every_nth_epoch, num_epochs)
        assert collect in set(
            ("cpu", "gpu")
        ), f"Collect device {collect} unknown."
        self.collect = collect
        self.output_dir = output_dir
        self.evaluator = evaluator
        self.logging_disabled = False
        self.run_eval = True

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def on_test_epoch_end(self, model: nn.Module, epoch: int) -> None:
        """Hook to run at the end of a testing epoch."""
        self.evaluator.gather(all_gather_object_cpu)
        if get_rank() == 0:
            self.evaluate()
        self.evaluator.reset()

    def on_test_batch_end(self, model: nn.Module, inputs: DictStrAny) -> None:
        """Hook to run at the end of a testing batch."""
        self.evaluator.process(**inputs)

    def evaluate(self) -> dict[str, MetricLogs]:
        """Evaluate the performance after processing all input/output pairs."""
        if not self.run_eval:
            return {}

        results = {}
        if not self.logging_disabled:
            rank_zero_info("Running evaluator %s...", str(self.evaluator))

        for metric in self.evaluator.metrics:
            if self.output_dir is not None:
                output_dir = os.path.join(self.output_dir, metric)
                os.makedirs(output_dir, exist_ok=True)
                # self.evaluator.t(output_dir, metric)  # TODO implement save

            log_dict, log_str = self.evaluator.evaluate(metric)
            results[metric] = log_dict
            if not self.logging_disabled:
                for k, v in log_dict.items():
                    rank_zero_info("%s: %.3f", k, v)
                rank_zero_info("Showing results for %s", metric)
                rank_zero_info(log_str)
        return results


class VisualizerCallback(Callback):
    """Callback for model visualization."""

    def __init__(
        self,
        visualizer: Visualizer,
        output_dir: None | str = None,
        collect: str = "cpu",
        run_every_nth_epoch: int = 1,
        num_epochs: int = -1,
    ) -> None:
        """Init callback.

        Args:
            visualizer (Visualizer): Visualizer.
            output_dir (str, Optional): Output directory for saving the
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
        self.output_dir = output_dir
        self.visualizer = visualizer

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def on_test_epoch_end(self, model: nn.Module, epoch: int) -> None:
        """Hook to run at the end of a testing epoch."""
        if get_rank() == 0:
            if self.output_dir is not None:
                self.visualizer.save_to_disk(self.output_dir)
        self.visualizer.reset()

    def on_test_batch_end(self, model: nn.Module, inputs: DictStrAny) -> None:
        """Hook to run at the end of a testing batch."""
        self.visualizer.process(**inputs)


class LoggingCallback(Callback):
    """Callback for logging."""

    def __init__(self, refresh_rate: int = 50) -> None:
        """Init callback."""
        super().__init__(1, -1)
        self._refresh_rate = refresh_rate
        self._metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.timer = Timer()

    def on_train_epoch_start(self, model: nn.Module, epoch: int) -> None:
        """Hook to run at the start of a training epoch."""
        self.timer.reset()
        rank_zero_info(f"Epoch {epoch + 1} started.")

    def on_train_batch_end(
        self,
        model: nn.Module,
        inputs: DictStrAny,
        metrics: DictStrAny,
        cur_epoch: int,
        total_epochs: int,
        cur_batch: int,
        total_batches: int,
    ) -> None:
        """Hook to run at the end of a training batch."""
        for k, v in metrics.items():
            self._metrics[k].append(v)
        if cur_batch % self._refresh_rate == self._refresh_rate - 1:
            rank_zero_info(
                compose_log_str(
                    f"Epoch {cur_epoch + 1}/{total_epochs}",
                    cur_batch + 1,
                    total_batches,
                    self.timer,
                    {
                        k: sum(v) / len(v) if len(v) > 0 else float("NaN")
                        for k, v in self._metrics.items()
                    },
                )
            )
            self._metrics = defaultdict(list)


class CheckpointCallback(Callback):
    """Callback for model checkpointing."""

    def __init__(
        self,
        save_prefix: str,
        run_every_nth_epoch: int = 1,
        num_epochs: int = -1,
    ) -> None:
        """Init callback.

        Args:
            save_prefix (str): Prefix of checkpoint path for saving.
            run_every_nth_epoch (int): Save model checkpoint every nth epoch.
                Defaults to 1.
            num_epochs (int): Number of total epochs, used for determining
                whether to visualize at the final epoch. Defaults to -1.
        """
        super().__init__(run_every_nth_epoch, num_epochs)
        self.save_prefix = save_prefix

    def on_train_epoch_end(self, model: nn.Module, epoch: int) -> None:
        """Hook to run at the end of a training epoch."""
        os.makedirs(os.path.dirname(self.save_prefix), exist_ok=True)
        torch.save(
            model.state_dict(),  # TODO, save full state dict with
            # optimizer, scheduler, etc.
            f"{self.save_prefix}/model_e{epoch + 1}.pt",
        )
