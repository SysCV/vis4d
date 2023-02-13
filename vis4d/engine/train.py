"""Vis4D trainer."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vis4d.common import DictStrAny
from vis4d.common.callbacks import Callback
from vis4d.common.logging import rank_zero_info
from vis4d.common.progress import compose_log_str
from vis4d.common.time import Timer
from vis4d.data import DictData
from vis4d.engine.connectors import DataConnector

from .opt import Optimizer
from .test import Tester
from .util import move_data_to_device


class Trainer:
    """Vis4D Trainer."""

    def __init__(
        self,
        num_epochs: int,
        log_step: int,
        dataloaders: DataLoader[DictData],
        data_connector: DataConnector,
        train_callbacks: dict[str, Callback] | None,
        test_every_nth_epoch: int = 1,
    ) -> None:
        """Initialize the trainer.

        Args:
            num_epochs (int): Number of training epochs.
            log_step (int): Interval for logging losses.
            dataloaders (DataLoader[DictData]): Dataloader for training.
            data_connector (DataConnector): Data connector used for generating
                training inputs from a batch of data.
            train_callbacks (dict[str, Callback] | None): Callback functions
                used during training.
            test_every_nth_epoch (int, optional): Interval for evaluating the
                model during training. Defaults to 1.
        """
        self.num_epochs = num_epochs
        self.log_step = log_step
        self.test_every_nth_epoch = test_every_nth_epoch
        self.train_dataloader = dataloaders
        self.data_connector = data_connector

        if train_callbacks is None:
            self.train_callbacks = {}
        else:
            self.train_callbacks = train_callbacks

        self.timer = Timer()

    def _run_test_on_epoch(self, epoch: int) -> bool:
        """Return whether to run test on current training epoch.

        Args:
            epoch (int): Current training epoch.

        Returns:
            bool: Whether to run test.
        """
        return epoch % self.test_every_nth_epoch == (
            self.test_every_nth_epoch - 1
        )

    def calculate_losses(
        self,
        cur_iter: int,
        epoch: int,
        output: DictData,
        data: DictData,
        loss: torch.nn.Module,
        running_losses: DictStrAny,
    ) -> None:
        """Compute losses for a batch of training data and log statistics.

        Args:
            cur_iter (int): Current training iteration.
            epoch (int): Current training epoch.
            output (DictData): Model output.
            data (DictData): Batch of training data.
            loss (torch.nn.Module): Loss function.
            running_losses (DictStrAny): Running statistics of losses.
        """
        loss_input = self.data_connector.get_loss_input(output, data)
        losses = loss(**loss_input)
        total_loss = sum(losses.values())
        total_loss.backward()

        # update statistics
        losses = {"loss": total_loss, **losses}
        for k, v in losses.items():
            if k in running_losses:
                running_losses[k] += v
            else:
                running_losses[k] = v

        # log losses
        if cur_iter % self.log_step == (self.log_step - 1):
            rank_zero_info(
                compose_log_str(
                    f"Epoch {epoch + 1}",
                    cur_iter + 1,
                    len(self.train_dataloader),
                    self.timer,
                    {k: v / self.log_step for k, v in running_losses.items()},
                )
            )

    def train(
        self,
        model: torch.nn.Module,
        optimizer: list[Optimizer],
        loss: torch.nn.Module | None = None,
        tester: None | Tester = None,
    ) -> None:
        """Training loop.

        Args:
            model: Model that should be trained.
            optimizer: Optimizer that should be used for training. This bundles
                the optimizer, the learning rate scheduler, and the warmup
                scheduler.
            loss: Loss function that should be used for training. Defaults to
                None.
            tester: Tester that should be used for testing. Defaults to None.
        """
        running_losses: DictStrAny = {}
        step = 0

        # Set up optimizers and schedulers. This is done here because the
        # optimizers require the model parameters.
        for opt in optimizer:
            opt.setup(model)

        device = next(model.parameters()).device  # model device

        for epoch in range(self.num_epochs):
            # Set model to train mode
            model.train()

            if hasattr(self.train_dataloader, "sampler") and isinstance(
                self.train_dataloader.sampler, DistributedSampler
            ):
                self.train_dataloader.sampler.set_epoch(epoch)

            for i, data in enumerate(self.train_dataloader):
                # zero grad optimziers
                for opt in optimizer:
                    opt.zero_grad()

                # input data
                data_moved: DictData = move_data_to_device(data, device)
                train_input = self.data_connector.get_train_input(data_moved)

                # forward + backward + optimize
                output = model(**train_input)

                if loss is not None:
                    # Do we want to support no loss?
                    # Idea is to allow the user to somewhat define a custom
                    # loss implementation in a custom optimizer.step()

                    self.calculate_losses(
                        i, epoch, output, data_moved, loss, running_losses
                    )

                for opt in optimizer:
                    opt.step(step)

                for _, callback in self.train_callbacks.items():
                    if callback.run_on_epoch(epoch):
                        callback.on_train_batch_end()

                step += 1

            for _, callback in self.train_callbacks.items():
                if callback.run_on_epoch(epoch):
                    callback.on_train_epoch_end(model, epoch)

            # testing
            if tester is not None and self._run_test_on_epoch(epoch):
                tester.test(model, epoch)

        rank_zero_info("Training done.")
