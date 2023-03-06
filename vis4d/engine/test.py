"""Vis4D tester."""
from __future__ import annotations

import pdb

import torch
from torch import nn
from torch.utils.data import DataLoader

from vis4d.common.callbacks import Callback
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
        self.test_dataloaders = dataloaders
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

        # run callbacks on test epoch begin
        for k, callback in self.test_callbacks.items():
            if callback.run_on_epoch(epoch):
                callback.on_test_epoch_start(model, epoch or 0)

        for test_loader in self.test_dataloaders:
            for cur_iter, data in enumerate(test_loader):
                total_iters = len(test_loader)

                # input data
                device = next(model.parameters()).device  # model device
                data = move_data_to_device(data, device)
                test_input = self.data_connector.get_test_input(data)

                # forward
                output = model(**test_input)

                for k, callback in self.test_callbacks.items():
                    if callback.run_on_epoch(epoch):
                        shared_clbk_kwargs = {
                            "epoch": epoch,
                            "cur_iter": cur_iter,
                            "total_iters": total_iters,
                        }
                        clbk_kwargs = self.data_connector.get_callback_input(
                            k, output, data, "test"
                        )
                        callback.on_test_batch_end(
                            model,
                            shared_clbk_kwargs,
                            clbk_kwargs,
                        )

        # run callbacks on test epoch end
        for k, callback in self.test_callbacks.items():
            if callback.run_on_epoch(epoch):
                callback.on_test_epoch_end(model, epoch)
