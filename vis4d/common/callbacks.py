"""This module contains utilities for callbacks."""
from __future__ import annotations

import logging
import os

import torch
from torch import nn

from vis4d.common import ArgsType, MetricLogs
from vis4d.common.distributed import get_rank
from vis4d.common.typing import DictStrAny
from vis4d.data.typing import DictData
from vis4d.engine.connectors import DataConnector
from vis4d.eval.base import Evaluator
from vis4d.vis.base import Visualizer


class Callback:
    """Base class for Vis4D Callbacks."""

    def run_on_epoch(self, epoch: int | None) -> bool:
        """Returns whether to run callback for current epoch (default True)."""
        return epoch is None or epoch >= 0

    def on_train_epoch_end(self, model: nn.Module, epoch: int) -> None:
        """Hook to run at the end of a training epoch.

        Args:
            model (nn.Module): Model that is being trained.
            epoch (int): Current training epoch.
        """

    def on_train_batch_end(self) -> None:
        """Hook to run at the end of a training batch."""

    def on_test_epoch_end(self) -> None:
        """Hook to run at the end of a testing epoch."""

    def on_test_batch_end(
        self, outputs: ArgsType, batch: ArgsType, key_name: str
    ) -> None:
        """Hook to run at the end of a testing batch.

        Args:
            outputs (ArgsType): Model predictions.
            batch (ArgsType): Testing input data.
            key_name (str): Key name used to extract data using the data
                connector.
        """


def default_eval_connector(
    mode: str,  # pylint: disable=unused-argument
    data: DictData,
    outputs: ArgsType,
) -> DictStrAny:
    """Default eval connector forwards input and outputs."""
    return {"data": data, "outputs": outputs}


class EvaluatorCallback(Callback):
    """Callback for model evaluation."""

    def __init__(
        self,
        evaluator: Evaluator,
        eval_connector: DataConnector,
        test_every_nth_epoch: int = 1,
        num_epochs: int = -1,
        output_dir: None | str = None,
        collect: str = "cpu",
    ) -> None:
        """Init callback.

        Args:
            evaluator (Evaluator): Evaluator.
            eval_connector (DataConnector): Data connector for evaluator.
            test_every_nth_epoch (int): Evaluate model every nth epoch.
                Defaults to 1.
            num_epochs (int): Number of total epochs, used for determining
                whether to evaluate at the final epoch. Defaults to -1.
            output_dir (str, Optional): Output directory for saving the
                evaluation results. Defaults to None (no save).
            collect (str): Which device to collect results across GPUs on.
                Defaults to "cpu".
        """
        assert collect in set(
            ("cpu", "gpu")
        ), f"Collect device {collect} unknown."
        self.collect = collect
        self.output_dir = output_dir
        self.evaluator = evaluator
        self.eval_connector = eval_connector
        self.test_every_nth_epoch = test_every_nth_epoch
        self.num_epochs = num_epochs
        self.logging_disabled = False
        self.run_eval = True

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def run_on_epoch(self, epoch: int | None) -> bool:
        """Returns whether to run callback for current epoch (default True)."""
        return (
            epoch is None
            or epoch == self.num_epochs - 1
            or epoch % self.test_every_nth_epoch
            == self.test_every_nth_epoch - 1
        )

    def on_test_epoch_end(self) -> None:
        """Hook to run at the end of a testing epoch."""
        # TODO: need an all_gather function?
        # def gather_func(x):
        #     return all_gather_object_cpu(x)

        # self.evaluator.gather(gather_func)
        if get_rank() == 0:
            self.evaluate()
        self.evaluator.reset()

    def on_test_batch_end(
        self, outputs: ArgsType, batch: ArgsType, key_name: str
    ) -> None:
        """Hook to run at the end of a testing batch."""
        # TODO, this should be all numpy.
        eval_inputs = self.eval_connector.get_evaluator_input(
            key_name, outputs, batch
        )
        self.evaluator.process(**eval_inputs)

    def evaluate(self) -> dict[str, MetricLogs]:
        """Evaluate the performance after processing all input/output pairs."""
        if not self.run_eval:
            return {}

        results = {}
        logger = logging.getLogger(__name__)
        if not self.logging_disabled:
            logger.info("Running evaluator %s...", str(self.evaluator))

        for metric in self.evaluator.metrics:
            if self.output_dir is not None:
                output_dir = os.path.join(self.output_dir, metric)
                os.makedirs(output_dir, exist_ok=True)
                # self.evaluator.t(output_dir, metric)  # TODO implement save

            log_dict, log_str = self.evaluator.evaluate(metric)
            results[metric] = log_dict
            if not self.logging_disabled:
                for k, v in log_dict.items():
                    self.log(k, v, rank_zero_only=True)  # type: ignore # pylint: disable=no-member,line-too-long
                logger.info("Showing results for %s", metric)
                logger.info(log_str)
        return results


class VisualizerCallback(Callback):
    """Callback for model visualization."""

    def __init__(
        self,
        visualizer: Visualizer,
        data_connector: DataConnector,
        vis_every_nth_epoch: int = 1,
        num_epochs: int = -1,
        output_dir: None | str = None,
        collect: str = "cpu",
    ) -> None:
        """Init callback.

        Args:
            visualizer (Visualizer): Visualizer.
            data_connector (DataConnector): Data connector for visualizer.
            vis_every_nth_epoch (int): Visualize results every nth epoch.
                Defaults to 1.
            num_epochs (int): Number of total epochs, used for determining
                whether to visualize at the final epoch. Defaults to -1.
            output_dir (str, Optional): Output directory for saving the
                visualizations. Defaults to None (no save).
            collect (str): Which device to collect results across GPUs on.
                Defaults to "cpu".
        """
        assert collect in set(
            ("cpu", "gpu")
        ), f"Collect device {collect} unknown."
        self.collect = collect
        self.output_dir = output_dir
        self.visualizer = visualizer
        self.data_connector = data_connector
        self.vis_every_nth_epoch = vis_every_nth_epoch
        self.num_epochs = num_epochs

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def run_on_epoch(self, epoch: int | None) -> bool:
        """Returns whether to run callback for current epoch (default True)."""
        return (
            epoch is None
            or epoch == self.num_epochs - 1
            or epoch % self.vis_every_nth_epoch == self.vis_every_nth_epoch - 1
        )

    def on_test_epoch_end(self) -> None:
        """Hook to run at the end of a testing epoch."""
        # TODO: need an all_gather function?
        # def gather_func(x):
        #     return all_gather_object_cpu(x)

        # self.evaluator.gather(gather_func)
        if get_rank() == 0:
            if self.output_dir is not None:
                self.visualizer.save_to_disk(self.output_dir)
        self.visualizer.reset()

    def on_test_batch_end(
        self, outputs: ArgsType, batch: ArgsType, key_name: str
    ) -> None:
        """Hook to run at the end of a testing batch."""
        eval_kwargs = self.data_connector.get_visualizer_input(
            key_name, outputs, batch
        )
        self.visualizer.process(**eval_kwargs)


class CheckpointCallback(Callback):
    """Callback for model checkpointing."""

    def __init__(
        self, save_prefix: str, save_every_nth_epoch: int = 1
    ) -> None:
        """Init callback.

        Args:
            save_prefix (str): Prefix of checkpoint path for saving.
            save_every_nth_epoch (int): Save model checkpoint every nth epoch.
                Defaults to 1.
        """
        self.save_prefix = save_prefix
        self.save_every_nth_epoch = save_every_nth_epoch

    def run_on_epoch(self, epoch: int | None) -> bool:
        """Returns whether to run callback for current epoch (default True)."""
        return epoch is None or epoch % self.save_every_nth_epoch == (
            self.save_every_nth_epoch - 1
        )

    def on_train_epoch_end(self, model: nn.Module, epoch: int) -> None:
        """Hook to run at the end of a training epoch."""
        os.makedirs(os.path.dirname(self.save_prefix), exist_ok=True)
        torch.save(
            model.state_dict(),  # TODO, save full state dict with
            # optimizer, scheduler, etc.
            f"{self.save_prefix}/model_e{epoch + 1}.pt",
        )
