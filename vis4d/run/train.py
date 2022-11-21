"""Vis4D trainer."""
from __future__ import annotations

from time import perf_counter

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vis4d.common.distributed import get_rank
from vis4d.eval import Evaluator
from vis4d.optim.warmup import BaseLRWarmup
from vis4d.vis.base import Visualizer

from .test import testing_loop
from .util import move_data_to_device


def training_loop(
    train_dataloader: DataLoader,
    test_dataloader: list[DataLoader],
    evaluators: list[Evaluator],
    metric: str,
    model: nn.Module,
    loss: nn.Module,
    data_connector,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    log_step: int,
    learning_rate: float,
    save_prefix: str,
    warmup: None | BaseLRWarmup = None,
    visualizers: tuple[Visualizer] = (),
    eval_connector=None,  # TODO, discuss
    test_every_nth_epoch=1,
    save_every_nth_epoch=1,
    vis_every_nth_epoch=1,
) -> None:
    """Training loop."""

    if eval_connector is None:
        # For now just wrap data connector to not break anything.
        eval_connector = lambda in_data, out_data: data_connector(in_data)

    running_losses = {}
    for epoch in range(num_epochs):
        model.train()
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        for i, data in enumerate(train_dataloader):
            tic = perf_counter()

            # zero the parameter gradients
            optimizer.zero_grad()
            # input data
            device = next(model.parameters()).device  # model device
            data = move_data_to_device(data, device)
            train_input = data_connector("train", data)
            loss_input = data_connector("loss", data)
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
            if i % log_step == (log_step - 1):
                log_str = f"[{epoch + 1}, {i + 1:5d} / ?] "
                for k, v in running_losses.items():
                    log_str += f"{k}: {v / log_step:.3f}, "
                print(log_str.rstrip(", "))  # FIXME move to log statement
                running_losses = {}
        scheduler.step()
        if (
            epoch % save_every_nth_epoch == (save_every_nth_epoch - 1)
            and get_rank() == 0
        ):
            torch.save(
                model.module.state_dict(), f"{save_prefix}_{epoch + 1}.pt"
            )
        # Make sure to test at last epoch or at desired frequency
        if (epoch == num_epochs - 1) or epoch % test_every_nth_epoch == (
            test_every_nth_epoch - 1
        ):
            # Visualize after last epoch or at requested frequency
            visualizers_to_use = (
                visualizers
                if (
                    epoch == num_epochs - 1
                    or epoch % vis_every_nth_epoch == vis_every_nth_epoch - 1
                )
                else []
            )
            testing_loop(
                test_dataloader,
                evaluators,
                metric,
                model,
                data_connector,
                eval_connector,
                visualizers_to_use,
            )
    print("training done.")  # FIXME move to log statement
