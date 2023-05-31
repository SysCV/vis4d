"""Trainer for running train and test."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vis4d.data import DictData
from vis4d.engine.callbacks import Callback, TrainerState
from vis4d.engine.connectors import DataConnector
from vis4d.engine.loss_module import LossModule

from .optim import Optimizer
from .util import move_data_to_device


class Trainer:
    """Trainer class."""

    def __init__(
        self,
        device: torch.device,
        train_dataloader: DataLoader[DictData] | None,
        test_dataloader: list[DataLoader[DictData]] | None,
        train_data_connector: DataConnector | None,
        test_data_connector: DataConnector | None,
        callbacks: list[Callback],
        num_epochs: int = 0,
        num_steps: int = 0,
        epoch: int = 0,
        global_step: int = 0,
        val_check_interval: int = -1,
    ) -> None:
        """Initialize the trainer.

        Args:
            device (torch.device): Device that should be used for training.
            train_dataloader (DataLoader[DictData] | None, optional):
                Dataloader for training.
            test_dataloader (list[DataLoader[DictData]] | None, optional):
                Dataloaders for testing.
            train_data_connector (DataConnector | None): Data connector used
                for generating training inputs from a batch of data.
            test_data_connector (DataConnector | None): Data connector used for
                generating testing inputs from a batch of data.
            callbacks (list[Callback]): Callbacks that should be used during
                training.
            num_epochs (int): Number of training epochs. Defaults to 0.
            num_steps (int): Number of training steps. Defaults to 0.
            epoch (int, optional): Starting epoch. Defaults to 0.
            global_step (int, optional): Starting step. Defaults to 0.
            val_check_interval (int, optional): Interval for evaluating the
                model during training. Defaults to -1.
        """
        self.device = device
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.train_data_connector = train_data_connector
        self.test_data_connector = test_data_connector
        self.callbacks = callbacks
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.val_check_interval = val_check_interval

        self.epoch = epoch
        self.global_step = global_step

        assert (self.num_epochs > 0) != (
            self.num_steps > 0
        ), "Either num_epochs or num_steps must be > 0, not none or both."
        if self.num_epochs > 0:
            self.epoch_based = True
        else:
            self.epoch_based = False
            self.num_epochs = 999
        if val_check_interval == -1:
            # default to running validation every epoch / 1000 steps
            self.val_check_interval = 1 if self.num_epochs > 0 else 1000

    def _run_test_on_iteration(self, iteration: int) -> bool:
        """Return whether to run test on current training iteration.

        Args:
            iteration (int): Current training iteration. Can be either epoch
                or step number.

        Returns:
            bool: Whether to run test.
        """
        return iteration % self.val_check_interval == (
            self.val_check_interval - 1
        )

    def fit(
        self,
        model: nn.Module,
        optimizers: list[Optimizer],
        loss_module: LossModule,
    ) -> None:
        """Training loop.

        Args:
            model (nn.Module): Model that should be trained.
            optimizers (list[Optimizer]): Optimizers that should be used for
                training. This bundles the optimizers, the learning rate
                schedulers, and the warmup schedulers.
            loss_module (LossModule): Loss module that should be used for
                training.

        Raises:
            TypeError: If the loss value is not a torch.Tensor or a dict of
                torch.Tensor.
        """
        assert (
            self.train_data_connector is not None
        ), "No train data connector."
        assert self.train_dataloader is not None, "No train dataloader."

        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch

            # Run callbacks for epoch begin
            for callback in self.callbacks:
                callback.on_train_epoch_start(self.get_state(), model)

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
                data = move_data_to_device(data, self.device)
                train_input = self.train_data_connector(data)

                # Forward + backward + optimize
                output = model(**train_input)

                losses = loss_module(output, data)

                metrics: dict[str, float] = {}
                if isinstance(losses, Tensor):
                    total_loss = losses
                elif isinstance(losses, dict):
                    total_loss = sum(losses.values())  # type: ignore
                    for k, v in losses.items():
                        metrics[k] = v.detach().cpu().item()
                else:
                    raise TypeError(
                        "Loss function must return a Tensor or a dict of "
                        + "Tensor"
                    )
                metrics["loss"] = total_loss.detach().cpu().item()

                total_loss.backward()

                for opt in optimizers:
                    opt.step_on_batch(self.global_step)

                for callback in self.callbacks:
                    callback.on_train_batch_end(
                        trainer_state=self.get_state(metrics),
                        model=model,
                        outputs=output,
                        batch=data,
                        batch_idx=batch_idx,
                    )

                # Testing (step-based)
                if not self.epoch_based:
                    if (
                        self._run_test_on_iteration(self.global_step)
                        and self.test_dataloader is not None
                    ):
                        self.test(model)

                    if self.global_step >= self.num_steps - 1:
                        break

                    # Set model back to train mode
                    model.train()

                self.global_step += 1

            # Update learning rate on epoch
            for opt in optimizers:
                opt.step_on_epoch(epoch)

            # Run callbacks for epoch end
            for callback in self.callbacks:
                callback.on_train_epoch_end(self.get_state(), model)

            # Testing (epoch-based)
            if self.epoch_based:
                if (
                    self._run_test_on_iteration(epoch)
                    and self.test_dataloader is not None
                ):
                    self.test(model)

            if self.global_step >= self.num_steps - 1:
                break

    @torch.no_grad()
    def test(self, model: nn.Module) -> None:
        """Testing loop.

        Args:
            model (nn.Module): Model that should be tested.
        """
        assert self.test_data_connector is not None, "No test data connector."
        assert self.test_dataloader is not None, "No test dataloader."

        model.eval()

        # run callbacks on test epoch begin
        for callback in self.callbacks:
            callback.on_test_epoch_start(self.get_state(), model)

        for i, test_loader in enumerate(self.test_dataloader):
            for batch_idx, data in enumerate(test_loader):
                data = move_data_to_device(data, self.device)
                test_input = self.test_data_connector(data)

                # forward
                output = model(**test_input)

                for callback in self.callbacks:
                    callback.on_test_batch_end(
                        trainer_state=self.get_state(),
                        model=model,
                        outputs=output,
                        batch=data,
                        batch_idx=batch_idx,
                        dataloader_idx=i,
                    )

        # run callbacks on test epoch end
        for callback in self.callbacks:
            callback.on_test_epoch_end(self.get_state(), model)

    def get_state(
        self, metrics: dict[str, float] | None = None
    ) -> TrainerState:
        """Get the state of the trainer."""
        num_train_batches = (
            len(self.train_dataloader)
            if self.train_dataloader is not None
            else None
        )

        num_test_batches = (
            [len(test_loader) for test_loader in self.test_dataloader]
            if self.test_dataloader is not None
            else None
        )

        trainer_state = TrainerState(
            current_epoch=self.epoch,
            num_epochs=self.num_epochs,
            global_step=self.global_step,
            num_steps=self.num_steps,
            train_dataloader=self.train_dataloader,
            num_train_batches=num_train_batches,
            test_dataloader=self.test_dataloader,
            num_test_batches=num_test_batches,
        )

        if metrics is not None:
            trainer_state["metrics"] = metrics

        return trainer_state
