"""Vis4D trainer."""
from __future__ import annotations

import logging
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vis4d.common import DictStrAny
from vis4d.common.distributed import get_rank
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
        test_every_nth_epoch: int = 1,
        save_every_nth_epoch: int = 1,
        # vis_every_nth_epoch: int = 1,
    ) -> None:
        """Creates an instance of the class."""
        self.num_epochs = num_epochs
        self.log_step = log_step
        self.test_every_nth_epoch = test_every_nth_epoch
        self.save_every_nth_epoch = save_every_nth_epoch
        self.train_dataloader = dataloaders
        self.data_connector = data_connector

        self.timer = Timer()

    def train(
        self,
        opt: Optimizer,
        save_prefix: str = "model",
        tester: None | Tester = None,
        metric: None | str = None,
    ) -> None:
        """Training loop."""
        logger = logging.getLogger(__name__)

        running_losses: DictStrAny = {}
        step = 0
        for epoch in range(self.num_epochs):
            opt.model.train()
            if hasattr(self.train_dataloader, "sampler") and isinstance(
                self.train_dataloader.sampler, DistributedSampler
            ):
                self.train_dataloader.sampler.set_epoch(epoch)
            for i, data in enumerate(self.train_dataloader):
                # zero the parameter gradients
                opt.optimizer.zero_grad()
                # input data
                device = next(opt.model.parameters()).device  # model device
                data_moved: DictData = move_data_to_device(data, device)
                train_input = self.data_connector.get_train_input(data_moved)
                # forward + backward + optimize
                output = opt.model(**train_input)
                loss_input = self.data_connector.get_loss_input(
                    output, train_input
                )

                losses = opt.loss(**loss_input)
                total_loss = sum(losses.values())
                total_loss.backward()

                opt.warmup_step(epoch)
                opt.optimizer.step()
                step += 1

                # print statistics
                losses = {"loss": total_loss, **losses}
                for k, v in losses.items():
                    if k in running_losses:
                        running_losses[k] += v
                    else:
                        running_losses[k] = v
                if i % self.log_step == (self.log_step - 1):
                    rank_zero_info(
                        compose_log_str(
                            f"Epoch {epoch + 1}",
                            i + 1,
                            len(self.train_dataloader),
                            self.timer,
                            {
                                k: v / self.log_step
                                for k, v in running_losses.items()
                            },
                        )
                    )

            if opt.lr_scheduler is not None:
                opt.lr_scheduler.step()

            if (
                epoch % self.save_every_nth_epoch
                == (self.save_every_nth_epoch - 1)
                and get_rank() == 0
            ):
                os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
                torch.save(
                    opt.model.state_dict(),
                    f"{save_prefix}_{epoch + 1}.pt",
                )

            # testing
            if tester is not None:
                if epoch % self.test_every_nth_epoch == (
                    self.test_every_nth_epoch - 1
                ):
                    assert metric is not None
                    tester.test(opt.model, metric, epoch)

        logger.info("training done.")
