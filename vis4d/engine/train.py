"""Vis4D trainer."""
from __future__ import annotations

import logging
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vis4d.common import DictStrAny
from vis4d.common.distributed import get_rank
from vis4d.data import DictData

from .opt import Optimizer
from .test import Tester
from .util import move_data_to_device


class Trainer:
    """Vis4D Trainer."""

    def __init__(
        self,
        num_epochs: int,
        log_step: int,
        test_every_nth_epoch: int = 1,
        save_every_nth_epoch: int = 1,
        vis_every_nth_epoch: int = 1,
    ) -> None:
        """Creates an instance of the class."""
        self.num_epochs = num_epochs
        self.log_step = log_step
        self.test_every_nth_epoch = test_every_nth_epoch
        self.save_every_nth_epoch = save_every_nth_epoch
        self.vis_every_nth_epoch = vis_every_nth_epoch

        self.train_dataloader = self.setup_train_dataloaders()

    def setup_train_dataloaders(self) -> DataLoader:
        """Set-up training data loaders."""
        raise NotImplementedError

    def data_connector(self, mode: str, data: DictData) -> DictData:
        """Connector between the data and the model."""
        assert mode in {"train", "loss"}
        return data

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
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)
            for i, data in enumerate(self.train_dataloader):
                tic = perf_counter()

                # zero the parameter gradients
                opt.optimizer.zero_grad()
                # input data
                device = next(opt.model.parameters()).device  # model device
                data = move_data_to_device(data, device)
                train_input = self.data_connector("train", data)
                loss_input = self.data_connector("loss", data)
                # forward + backward + optimize
                output = opt.model(**train_input)
                losses = opt.loss(output, **loss_input)
                total_loss = sum(losses.values())
                total_loss.backward()

                opt.warmup_step(epoch)
                opt.optimizer.step()
                step += 1
                toc = perf_counter()

                # print statistics
                losses = dict(time=toc - tic, loss=total_loss, **losses)
                for k, v in losses.items():
                    if k in running_losses:
                        running_losses[k] += v
                    else:
                        running_losses[k] = v
                if i % self.log_step == (self.log_step - 1):
                    log_str = f"[{epoch + 1}, {i + 1:5d} / ?] "
                    for k, v in running_losses.items():
                        log_str += f"{k}: {v / self.log_step:.3f}, "
                    logger.info(log_str.rstrip(", "))
                    running_losses = {}
            opt.lr_scheduler.step()

            if (
                epoch % self.save_every_nth_epoch
                == (self.save_every_nth_epoch - 1)
                and get_rank() == 0
            ):
                torch.save(
                    opt.model.module.state_dict(),
                    f"{save_prefix}_{epoch + 1}.pt",
                )

            # testing
            if tester is not None:
                assert metric is not None
                tester.test(opt.model, metric, epoch)
        logger.info("training done.")
