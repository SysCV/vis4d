"""Vis4D trainer."""
from __future__ import annotations

import logging
from time import perf_counter

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from vis4d.common import DictStrAny
from vis4d.common.distributed import get_rank
from vis4d.data import DictData
from vis4d.eval import Evaluator
from vis4d.optim.warmup import BaseLRWarmup
from vis4d.vis.base import Visualizer

from .test import testing_loop
from .util import move_data_to_device


class Trainer:
    """Vis4D Trainer."""

    # def __init__(self, TrainingLoop = DefaultTrainingLoop(), InferenceLoop, DataModule = DetectDataModule) -> None:
    def __init__(
        self,
        num_epochs: int,
        log_step: int,
        test_every_nth_epoch: int = 1,
        save_every_nth_epoch: int = 1,
        vis_every_nth_epoch: int = 1,
    ) -> None:
        """Init."""
        self.num_epochs = num_epochs
        self.log_step = log_step
        self.test_every_nth_epoch = test_every_nth_epoch
        self.save_every_nth_epoch = save_every_nth_epoch
        self.vis_every_nth_epoch = vis_every_nth_epoch

        self.train_dataloader = self.setup_train_dataloaders()
        self.test_dataloader = self.setup_test_dataloaders()

    def setup_train_dataloaders(self) -> DataLoader:
        """Set-up training data loaders."""
        raise NotImplementedError

    def setup_test_dataloaders(self) -> list[DataLoader]:
        """Set-up testing data loaders."""
        raise NotImplementedError

    def data_connector(self, mode: str, data: DictData) -> DictData:
        """Connector between the data and the model."""
        return data

    def setup_evaluators(self) -> list[Evaluator]:
        """Set-up evaluators."""
        raise NotImplementedError

    def evaluator_connector(
        self, data: DictData, output: DictStrAny
    ) -> DictStrAny:
        """Connector between the data and the evaluator."""
        # For now just wrap data connector to not break anything.
        return data

    def setup_visualizers(self) -> list[Visualizer]:
        """Set-up visualizers."""
        raise NotImplementedError

    def train(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        learning_rate: float,
        warmup: None | BaseLRWarmup = None,
        save_prefix: str = "model",
        metric: None | str = None,
    ) -> None:
        """Training loop."""
        logger = logging.getLogger(__name__)

        running_losses: DictStrAny = {}
        for epoch in range(self.num_epochs):
            model.train()
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)
            for i, data in enumerate(self.train_dataloader):
                tic = perf_counter()

                # zero the parameter gradients
                optimizer.zero_grad()
                # input data
                device = next(model.parameters()).device  # model device
                data = move_data_to_device(data, device)
                train_input = self.data_connector("train", data)
                loss_input = self.data_connector("loss", data)
                # forward + backward + optimize
                output = model(**train_input)
                losses = loss(output, **loss_input)
                total_loss = sum(losses.values())
                total_loss.backward()

                if warmup is not None:
                    if epoch == 0 and i < 500:
                        for g in optimizer.param_groups:
                            g["lr"] = warmup(i, learning_rate)
                    elif epoch == 0 and i == 500:
                        for g in optimizer.param_groups:
                            g["lr"] = learning_rate

                optimizer.step()
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
            scheduler.step()

            if (
                epoch % self.save_every_nth_epoch
                == (self.save_every_nth_epoch - 1)
                and get_rank() == 0
            ):
                torch.save(
                    model.module.state_dict(), f"{save_prefix}_{epoch + 1}.pt"
                )
            # Make sure to test at last epoch or at desired frequency
            if (
                epoch == self.num_epochs - 1
            ) or epoch % self.test_every_nth_epoch == (
                self.test_every_nth_epoch - 1
            ):
                # Visualize after last epoch or at requested frequency
                # visualizers_to_use = (
                #     visualizers
                #     if (
                #         epoch == self.num_epochs - 1
                #         or epoch % self.vis_every_nth_epoch
                #         == self.vis_every_nth_epoch - 1
                #     )
                #     else []
                # )
                assert metric is not None
                self.test(model, metric)
        logger.info("training done.")

    @torch.no_grad()
    def test(self, model: nn.Module, metric: str) -> None:
        """Testing loop."""
        logger = logging.getLogger(__name__)

        evaluators = self.setup_evaluators()
        visualizers = self.setup_visualizers()

        model.eval()
        logger.info("Running validation...")
        for test_loader in self.test_dataloader:
            for _, data in enumerate(tqdm(test_loader)):
                # input data
                device = next(model.parameters()).device  # model device
                data = move_data_to_device(data, device)
                test_input = self.data_connector("test", data)

                # forward
                output = model(**test_input)

                for test_eval in evaluators:
                    evaluator_kwargs = self.evaluator_connector(data, output)
                    test_eval.process(
                        *[
                            v.detach().cpu().numpy()
                            for k, v in evaluator_kwargs.items()
                        ]
                    )
                for vis in visualizers:
                    vis.process(data, output)
        for test_eval in evaluators:
            _, log_str = test_eval.evaluate(metric)
            logger.info(log_str)

        for test_vis in visualizers:
            test_vis.visualize()
            # test_vis.clear()
