"""Vis4D trainer."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vis4d.common.time import Timer
from vis4d.common.callbacks import Callback
from vis4d.common.logging import rank_zero_info
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
        grad_norm_clip: float | None = None,
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
        self.grad_norm_clip = grad_norm_clip

        if train_callbacks is None:
            self.train_callbacks = {}
        else:
            self.train_callbacks = train_callbacks

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

    def train(
        self,
        model: torch.nn.Module,
        optimizers: list[Optimizer],
        loss: torch.nn.Module | None = None,
        tester: None | Tester = None,
    ) -> None:
        """Training loop.

        Args:
            model: Model that should be trained.
            optimizers: Optimizers that should be used for training. This
                bundles the optimizers, the learning rate schedulers, and the
                warmup schedulers.
            loss: Loss function that should be used for training. Defaults to
                None.
            tester: Tester that should be used for testing. Defaults to None.

        Raises:
            TypeError: If the loss value is not a torch.Tensor or a dict of
                torch.Tensor.
        """
        step = 0

        # Set up optimizers and schedulers. This is done here because the
        # optimizers require the model parameters.
        for opt in optimizers:
            opt.setup(model)

        device = next(model.parameters()).device  # model device

        for epoch in range(self.num_epochs):
            total_iters = len(self.train_dataloader)

            # Run callbacks for epoch begin
            for _, callback in self.train_callbacks.items():
                if callback.run_on_epoch(epoch):
                    callback.on_train_epoch_begin(model, epoch)

            # Set model to train mode
            model.train()

            # Update learning rate on epoch
            for opt in optimizers:
                opt.step_on_epoch(epoch)

            # Set epoch for distributed sampler
            if hasattr(self.train_dataloader, "sampler") and isinstance(
                self.train_dataloader.sampler, DistributedSampler
            ):
                self.train_dataloader.sampler.set_epoch(epoch)

            # Training loop for one epoch
            for cur_iter, data in enumerate(self.train_dataloader):
                # Zero grad optimizers
                for opt in optimizers:
                    opt.zero_grad()

                # Input data
                data_moved: DictData = move_data_to_device(data, device)
                train_input = self.data_connector.get_train_input(data_moved)

                # Forward + backward + optimize
                output = model(**train_input)
                if loss is not None:
                    # Do we want to support no loss?
                    # Idea is to allow the user to somewhat define a custom
                    # loss implementation in a custom optimizer.step()
                    loss_input = self.data_connector.get_loss_input(
                        output, data_moved
                    )
                    losses = loss(**loss_input)
                    if isinstance(losses, torch.Tensor):
                        total_loss = losses
                        metrics = {"loss": losses}
                    elif isinstance(losses, dict):
                        total_loss = sum(losses.values())  # type: ignore
                        metrics = {"loss": total_loss, **losses}
                    else:
                        raise TypeError(
                            "Loss function must return a torch.Tensor or a "
                            "dict of torch.Tensors"
                        )
                    total_loss.backward()
                else:
                    losses = {}

                # clip gradients (if needed)
                if self.grad_norm_clip:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.grad_norm_clip
                    )

                for j, opt in enumerate(optimizers):
                    opt.step_on_batch(step)
                    try:
                        current_lr = opt.optimizer.param_groups[0]['lr']
                        losses[f"opt{j}_lr"] = current_lr
                    except:
                        pass

                for k, callback in self.train_callbacks.items():
                    if callback.run_on_epoch(epoch):
                        clbk_kwargs = self.data_connector.get_callback_input(
                            k, output, data_moved, "train"
                        )
                        callback.on_train_batch_end(
                            model,
                            clbk_kwargs,
                            metrics,
                            epoch,
                            self.num_epochs,
                            cur_iter,
                            total_iters,
                        )
                step += 1

            # Run callbacks for epoch end
            for _, callback in self.train_callbacks.items():
                if callback.run_on_epoch(epoch):
                    callback.on_train_epoch_end(model, epoch)

            # Testing
            if tester is not None and self._run_test_on_epoch(epoch):
                tester.test(model, epoch)

        rank_zero_info("Training done.")
