"""Vis4D tester."""
from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vis4d.common.callbacks import (
    Callback,
    EvaluatorCallback,
    VisualizerCallback,
)
from vis4d.data import DictData
from vis4d.engine.connectors import DataConnector
from vis4d.eval import Evaluator
from vis4d.vis.base import Visualizer

from .util import move_data_to_device


class Tester:
    """Vis4D Tester."""

    def __init__(
        self,
        dataloaders: list[DataLoader[DictData]],
        data_connector: DataConnector,
        evaluators: dict[str, Evaluator] | None = None,
        visualizers: dict[str, Visualizer] | None = None,
        num_epochs: int = -1,
        test_every_nth_epoch: int = 1,
        vis_every_nth_epoch: int = 1,
    ) -> None:
        """Creates an instance of the class."""
        self.test_dataloader = dataloaders
        self.data_connector = data_connector

        if evaluators is None:
            evaluators = {}
        eval_callbacks = {
            k: EvaluatorCallback(
                v, self.data_connector, test_every_nth_epoch, num_epochs
            )
            for k, v in evaluators.items()
        }
        if visualizers is None:
            visualizers = {}
        vis_callbacks = {
            k: VisualizerCallback(
                v,
                self.data_connector,
                vis_every_nth_epoch,
                num_epochs,
                "vis4d-workspace/test",
            )
            for k, v in visualizers.items()
        }
        self.callbacks: dict[str, Callback] = {
            **eval_callbacks,
            **vis_callbacks,
        }

    @torch.no_grad()  # type: ignore
    def test(
        self, model: nn.Module, metric: str, epoch: None | int = None
    ) -> None:
        """Testing loop."""
        logger = logging.getLogger(__name__)

        model.eval()
        logger.info("Running validation...")
        for test_loader in self.test_dataloader:
            for _, data in enumerate(tqdm(test_loader)):
                # input data
                device = next(model.parameters()).device  # model device
                data = move_data_to_device(data, device)
                test_input = self.data_connector.get_test_input(data)

                # forward
                output = model(**test_input)

                for k, callback in self.callbacks.items():
                    if callback.run_on_epoch(epoch):
                        callback.on_test_batch_end(output, data, k)

        for k, callback in self.callbacks.items():
            if callback.run_on_epoch(epoch):
                callback.on_test_epoch_end()
