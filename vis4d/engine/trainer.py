"""Trainer for running train and test."""
from __future__ import annotations

import torch

from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vis4d.common.callbacks import Callback
from vis4d.data import DictData
from vis4d.engine.connectors import DataConnector

from .optim import Optimizer
from .util import move_data_to_device


class Trainer:
    """Trainer class."""

    def __init__(
        self,
        num_epochs: int,
        data_connector: DataConnector,
        train_dataloader: DataLoader[DictData] | None = None,
        test_dataloader: list[DataLoader[DictData]] | None = None,
        callbacks: list[Callback] = [],
        epoch: int = 0,
        global_steps: int = 0,
        check_val_every_n_epoch: int = 1,
        num_train_batches: int | None = None,
        num_test_batches: list[int | None] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            num_epochs (int): Number of training epochs.
            dataloaders (DataLoader[DictData]): Dataloader for training.
            data_connector (DataConnector): Data connector used for generating
                training inputs from a batch of data.
            train_dataloader (DataLoader[DictData] | None, optional):
                Dataloader for training. Defaults to None.
            test_dataloader (list[DataLoader[DictData]] | None, optional):
                Dataloaders for testing. Defaults to None.
            callbacks (list[Callback], optional): Callbacks that should be
                used during training. Defaults to [].
            epoch (int, optional): Starting epoch. Defaults to 0.
            global_steps (int, optional): Starting steps. Defaults to 0.
            check_val_every_n_epoch (int, optional): Interval for evaluating
                the model during training. Defaults to 1.
        """
        self.num_epochs = num_epochs
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.data_connector = data_connector
        self.train_dataloader = train_dataloader
        self.num_train_batches = (
            len(train_dataloader)
            if num_train_batches is None
            else num_train_batches
        )
        self.test_dataloader = test_dataloader
        self.num_test_batches = (
            [len(dataloader) for dataloader in test_dataloader]
            if num_test_batches is None
            else num_test_batches
        )
        self.callbacks = callbacks

        self.epoch = epoch
        self.global_steps = global_steps
        self.metrics: dict[str, Tensor] = {}

    def _run_test_on_epoch(self, epoch: int) -> bool:
        """Return whether to run test on current training epoch.

        Args:
            epoch (int): Current training epoch.

        Returns:
            bool: Whether to run test.
        """
        return epoch % self.check_val_every_n_epoch == (
            self.check_val_every_n_epoch - 1
        )

    def fit(
        self, model: nn.Module, optimizers: list[Optimizer], loss: nn.Module
    ) -> None:
        """Training loop.

        Args:
            model (nn.Module): Model that should be trained.
            optimizers (list[Optimizer]): Optimizers that should be used for
                training. This bundles the optimizers, the learning rate
                schedulers, and the warmup schedulers.
            loss (nn.Module): Loss function that should be used for training.

        Raises:
            TypeError: If the loss value is not a torch.Tensor or a dict of
                torch.Tensor.
        """
        assert self.train_dataloader is not None, "No train dataloader."

        device = next(model.parameters()).device  # model device

        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch

            # Run callbacks for epoch begin
            for callback in self.callbacks:
                if callback.run_on_epoch(epoch):
                    callback.on_train_epoch_start(self, model)

            # Set model to train mode
            model.train()

            # Set epoch for distributed sampler
            if hasattr(self.train_dataloader, "sampler") and isinstance(
                self.train_dataloader.sampler, DistributedSampler
            ):
                self.train_dataloader.sampler.set_epoch(epoch)

            # Training loop for one epoch
            for batch_idx, data in enumerate(self.train_dataloader):
                # Zero grad optimizers
                for opt in optimizers:
                    opt.zero_grad()

                # Input data
                data = move_data_to_device(data, device)
                train_input = self.data_connector.get_train_input(data)

                # Forward + backward + optimize
                output = model(**train_input)

                loss_input = self.data_connector.get_loss_input(output, data)
                losses = loss(**loss_input)
                if isinstance(losses, Tensor):
                    total_loss = losses
                    self.metrics["loss"] = total_loss
                elif isinstance(losses, dict):
                    total_loss = sum(losses.values())  # type: ignore
                    self.metrics["loss"] = total_loss
                    self.metrics.update(losses)
                else:
                    raise TypeError(
                        "Loss function must return a Tensor or a dict of "
                        + "Tensor"
                    )
                total_loss.backward()

                for opt in optimizers:
                    opt.step_on_batch(self.global_steps)

                for callback in self.callbacks:
                    if callback.run_on_epoch(epoch):
                        callback.on_train_batch_end(
                            trainer=self,
                            model=model,
                            outputs=output,
                            batch=data,
                            batch_idx=batch_idx,
                        )

                self.global_steps += 1

            # Update learning rate on epoch
            for opt in optimizers:
                opt.step_on_epoch(epoch)

            # Run callbacks for epoch end
            for callback in self.callbacks:
                if callback.run_on_epoch(epoch):
                    callback.on_train_epoch_end(self, model)

            # Testing
            if (
                self._run_test_on_epoch(epoch)
                and self.test_dataloader is not None
            ):
                self.test(model, epoch)

    @torch.no_grad()
    def test(self, model: nn.Module, epoch: int = 0) -> None:
        """Testing loop.

        Args:
            model (nn.Module): Model that should be tested.
            test_dataloaders (list[DataLoader[DictData]]): Dataloaders for
                testing.
            epoch (int, optional): Epoch for testing. (0 if not used during
                training). Defaults to 0.
        """
        assert self.test_dataloader is not None, "No test dataloader."

        model.eval()

        # run callbacks on test epoch begin
        for callback in self.callbacks:
            if callback.run_on_epoch(epoch):
                callback.on_test_epoch_start(self, model)

        for i, test_loader in enumerate(self.test_dataloader):
            for batch_idx, data in enumerate(test_loader):
                # input data
                device = next(model.parameters()).device
                data = move_data_to_device(data, device)
                test_input = self.data_connector.get_test_input(data)

                # forward
                output = model(**test_input)

                for callback in self.callbacks:
                    if callback.run_on_epoch(epoch):
                        callback.on_test_batch_end(
                            trainer=self,
                            model=model,
                            outputs=output,
                            batch=data,
                            batch_idx=batch_idx,
                            dataloader_idx=i,
                        )

        # run callbacks on test epoch end
        for callback in self.callbacks:
            if callback.run_on_epoch(epoch):
                callback.on_test_epoch_end(self, model)
