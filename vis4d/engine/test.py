"""Vis4D tester."""
from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vis4d.common.callbacks import Callback
from vis4d.data import DictData
from vis4d.engine.connectors import DataConnector

from .util import move_data_to_device


class Tester:
    """Vis4D Tester."""

    def __init__(
        self,
        dataloaders: list[DataLoader[DictData]],
        data_connector: DataConnector,
        test_callbacks: dict[str, Callback] | None,
    ) -> None:
        """Creates an instance of the class."""
        self.test_dataloader = dataloaders
        self.data_connector = data_connector

        if test_callbacks is None:
            self.test_callbacks = {}
        else:
            self.test_callbacks = test_callbacks

    @torch.no_grad()  # type: ignore
    def test(self, model: nn.Module, epoch: None | int = None) -> None:
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

                for k, callback in self.test_callbacks.items():
                    if callback.run_on_epoch(epoch):
                        callback.on_test_batch_end(output, data, k)

        for k, callback in self.test_callbacks.items():
            if callback.run_on_epoch(epoch):
                callback.on_test_epoch_end()
