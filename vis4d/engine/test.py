"""Vis4D tester."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vis4d.common.callbacks import Callback
from vis4d.common.logging import rank_zero_info
from vis4d.data import DictData
from vis4d.engine.connectors import DataConnector

from .util import move_data_to_device


class Tester:
    """Vis4D Tester."""

    def __init__(
        self,
        dataloaders: dict[str, DataLoader[DictData]],
        data_connector: DataConnector,
        test_callbacks: dict[str, Callback] | None,
    ) -> None:
        """Initialize the tester.

        Args:
            dataloaders (dict[str, DataLoader[DictData]]): Dataloaders for
                testing.
            data_connector (DataConnector): Data connector used for generating
                testing inputs from a batch of data.
            test_callbacks (dict[str, Callback] | None): Callback functions
                used during testing.
        """
        self.test_dataloader = dataloaders
        self.data_connector = data_connector

        if test_callbacks is None:
            self.test_callbacks = {}
        else:
            self.test_callbacks = test_callbacks

    @torch.no_grad()
    def test(self, model: nn.Module, epoch: None | int = None) -> None:
        """Testing loop.

        Args:
            model (nn.Module): Model that should be tested.
            epoch (None | int, optional): Epoch for testing (None if not used
                during training). Defaults to None.
        """
        model.eval()
        rank_zero_info("Running validation...")
        for dl_k, test_loader in self.test_dataloader.items():
            for _, data in enumerate(tqdm(test_loader, mininterval=10.0)):
                # input data
                device = next(model.parameters()).device  # model device
                data = move_data_to_device(data, device)
                test_input = self.data_connector.get_test_input(data)

                # forward
                output = model(**test_input)

                for k, callback in self.test_callbacks.items():
                    if dl_k == k and callback.run_on_epoch(epoch):
                        callback.on_test_batch_end(
                            model,
                            self.data_connector.get_callback_input(
                                k, output, data, "test"
                            ),
                        )

        for k, callback in self.test_callbacks.items():
            if callback.run_on_epoch(epoch):
                callback.on_test_epoch_end(model, epoch or 0)
