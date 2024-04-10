"""Trainer for running train and test."""

from __future__ import annotations

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter

from vis4d.common.distributed import rank_zero_only
from vis4d.common.logging import rank_zero_info, rank_zero_warn
from vis4d.data.typing import DictData
from vis4d.engine.callbacks import Callback, TrainerState
from vis4d.engine.connectors import DataConnector
from vis4d.engine.loss_module import LossModule

from .optim import LRSchedulerWrapper
from .util import move_data_to_device


class Trainer:
    """Trainer class."""

    def __init__(
        self,
        device: torch.device,
        output_dir: str,
        train_dataloader: DataLoader[DictData] | None,
        test_dataloader: list[DataLoader[DictData]] | None,
        train_data_connector: DataConnector | None,
        test_data_connector: DataConnector | None,
        callbacks: list[Callback],
        num_epochs: int = 1000,
        num_steps: int = -1,
        epoch: int = 0,
        global_step: int = 0,
        check_val_every_n_epoch: int | None = 1,
        val_check_interval: int | None = None,
        log_every_n_steps: int = 50,
    ) -> None:
        """Initialize the trainer.

        Args:
            device (torch.device): Device that should be used for training.
            output_dir (str): Output directory for saving tensorboard logs.
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
            num_epochs (int, optional): Number of training epochs. Defaults to
                1000.
            num_steps (int, optional): Number of training steps. Defaults to
                -1.
            epoch (int, optional): Starting epoch. Defaults to 0.
            global_step (int, optional): Starting step. Defaults to 0.
            check_val_every_n_epoch (int | None, optional): Evaluate the model
                every n epochs during training. Defaults to 1.
            val_check_interval (int | None, optional): Interval for evaluating
                the model during training. Defaults to None.
            log_every_n_steps (int, optional): Log the training status every n
                steps. Defaults to 50.
        """
        self.device = device
        self.output_dir = output_dir
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.train_data_connector = train_data_connector
        self.test_data_connector = test_data_connector
        self.callbacks = callbacks

        if num_epochs == -1 and num_steps == -1:
            rank_zero_info(
                "Neither `num_epochs` nor `num_steps` is specified. "
                + "Training will run indefinitely."
            )

        self.num_epochs = num_epochs
        self.num_steps = num_steps

        if check_val_every_n_epoch is None and val_check_interval is None:
            rank_zero_warn("Validation is disabled during training.")

        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.val_check_interval = val_check_interval
        self.log_every_n_steps = log_every_n_steps

        self.epoch = epoch
        self.global_step = global_step

        self._setup_logger()

    @rank_zero_only
    def _setup_logger(self) -> None:
        """Setup trainer logger."""
        self.writer = SummaryWriter(self.output_dir)

    @rank_zero_only
    def _teardown_logger(self) -> None:
        """Teardown trainer logger."""
        self.writer.close()

    @rank_zero_only
    def _log_scalar(self, tag: str, scalar_value: float) -> None:
        """Setup trainer logger."""
        self.writer.add_scalar(tag, scalar_value, self.global_step)

    def _log_lr(self, optimizer: Optimizer) -> None:
        """Log learning rate."""
        tag = f"lr-{optimizer.__class__.__name__}"

        if len(optimizer.param_groups) > 1:
            for i, param_group in enumerate(optimizer.param_groups):
                self._log_scalar(f"{tag}/pg{i+1}", param_group["lr"])
        else:
            self._log_scalar(tag, optimizer.param_groups[0]["lr"])

    def _run_test_on_epoch(self, epoch: int) -> bool:
        """Return whether to run test on current training epoch.

        Args:
            epoch (int): Current training epoch.

        Returns:
            bool: Whether to run test.
        """
        if self.check_val_every_n_epoch is None:
            return False
        return (epoch + 1) % self.check_val_every_n_epoch == 0

    def _run_test_on_step(self, step: int) -> bool:
        """Return whether to run test on current training step.

        Args:
            step (int): Current training step.

        Returns:
            bool: Whether to run test.
        """
        if self.val_check_interval is None:
            return False
        return (step + 1) % self.val_check_interval == 0

    def done(self) -> bool:
        """Return whether training is done."""
        is_done = False
        if _is_max_limit_reached(self.global_step, self.num_steps):
            rank_zero_info(
                f"`Trainer.fit` stopped: `num_steps={self.num_steps!r}` "
                + "reached."
            )
            is_done = True
        elif _is_max_limit_reached(self.epoch, self.num_epochs):
            rank_zero_info(
                f"`Trainer.fit` stopped: `num_epochs={self.num_epochs!r}` "
                + "reached."
            )
            is_done = True

        if is_done:
            self._teardown_logger()

        return is_done

    def fit(
        self,
        model: nn.Module,
        optimizers: list[Optimizer],
        lr_schedulers: list[LRSchedulerWrapper],
        loss_module: LossModule,
    ) -> None:
        """Training loop.

        Args:
            model (nn.Module): Model that should be trained.
            optimizers (list[Optimizer]): Optimizers that should be used for
                training.
            lr_schedulers (list[LRSchedulerWrapper]): Learning rate schedulers
                that should be used for training.
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

        while True:
            # Run callbacks for epoch begin
            for callback in self.callbacks:
                callback.on_train_epoch_start(
                    self.get_state(), model, loss_module
                )

            # Set model to train mode
            model.train()

            # Set epoch for distributed sampler
            if hasattr(self.train_dataloader, "sampler") and isinstance(
                self.train_dataloader.sampler, DistributedSampler
            ):
                self.train_dataloader.sampler.set_epoch(self.epoch)

            # Training loop for one epoch
            for batch_idx, data in enumerate(self.train_dataloader):
                # Log epoch
                if (self.global_step + 1) % self.log_every_n_steps == 0:
                    self._log_scalar("epoch", self.epoch)

                # Zero grad optimizers
                for opt in optimizers:
                    opt.zero_grad()

                # Input data
                data = move_data_to_device(data, self.device)

                for callback in self.callbacks:
                    callback.on_train_batch_start(
                        trainer_state=self.get_state(),
                        model=model,
                        loss_module=loss_module,
                        batch=data,
                        batch_idx=batch_idx,
                    )

                # Forward + backward + optimize
                output = model(**self.train_data_connector(data))

                total_loss, metrics = loss_module(output, data)

                total_loss.backward()

                for optimizer in optimizers:
                    # Log learning rate
                    if (self.global_step + 1) % self.log_every_n_steps == 0:
                        self._log_lr(optimizer)

                    # Step optimizers
                    optimizer.step()
                    self.global_step += 1

                # Step learning rate schedulers
                for lr_scheduler in lr_schedulers:
                    lr_scheduler.step_on_batch(self.global_step)

                for callback in self.callbacks:
                    log_dict = callback.on_train_batch_end(
                        trainer_state=self.get_state(metrics),
                        model=model,
                        loss_module=loss_module,
                        outputs=output,
                        batch=data,
                        batch_idx=batch_idx,
                    )

                    if log_dict is not None:
                        for k, v in log_dict.items():
                            self._log_scalar(f"train/{k}", v)

                # Testing (step-based)
                if (
                    self._run_test_on_step(self.global_step)
                    and self.test_dataloader is not None
                ):
                    self.test(model)

                    # Set model back to train mode
                    model.train()

                if self.done():
                    return

            # Update learning rate on epoch
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step(self.epoch)

            # Run callbacks for epoch end
            for callback in self.callbacks:
                callback.on_train_epoch_end(
                    self.get_state(
                        optimizers=optimizers, lr_schedulers=lr_schedulers
                    ),
                    model,
                    loss_module,
                )

            # Testing (epoch-based)
            if (
                self._run_test_on_epoch(self.epoch)
                and self.test_dataloader is not None
            ):
                self.test(model, is_val=True)

            self.epoch += 1

            if self.done():
                return

    @torch.no_grad()
    def test(self, model: nn.Module, is_val: bool = False) -> None:
        """Testing loop.

        Args:
            model (nn.Module): Model that should be tested.
            is_val (bool): Whether the test is run on the validation set during
                training.
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
            log_dict = callback.on_test_epoch_end(self.get_state(), model)

            if log_dict is not None:
                for k, v in log_dict.items():
                    key = f"val/{k}" if is_val else f"test/{k}"
                    self._log_scalar(key, v)

    def get_state(
        self,
        metrics: dict[str, float] | None = None,
        optimizers: list[Optimizer] | None = None,
        lr_schedulers: list[LRSchedulerWrapper] | None = None,
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
            train_module=self,
            train_engine="vis4d",
        )

        if metrics is not None:
            trainer_state["metrics"] = metrics

        if optimizers is not None:
            trainer_state["optimizers"] = optimizers

        if lr_schedulers is not None:
            trainer_state["lr_schedulers"] = lr_schedulers

        return trainer_state


def _is_max_limit_reached(current: int, maximum: int = -1) -> bool:
    """Check if the limit has been reached (if enabled).

    Args:
        current: the current value
        maximum: the maximum value (or -1 to disable limit)

    Returns:
        bool: whether the limit has been reached
    """
    return maximum != -1 and current >= maximum
