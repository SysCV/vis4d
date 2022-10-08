"""Vis4D trainer."""
from time import perf_counter
from typing import List, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from vis4d.eval import Evaluator
from vis4d.optim.warmup import BaseLRWarmup

from .test import testing_loop
from .util import move_data_to_device


def training_loop(
    train_dataloader: DataLoader,
    test_dataloader: List[DataLoader],
    evaluators: List[Evaluator],
    metric: str,
    model: nn.Module,
    loss: nn.Module,
    model_train_keys: List[str],
    model_test_keys: List[str],
    loss_keys: List[str],
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    log_step: int,
    learning_rate: float,
    save_prefix: str,
    warmup: Optional[BaseLRWarmup] = None,
) -> None:
    """Training loop."""
    running_losses = {}
    for epoch in range(num_epochs):
        model.train()
        for i, data in enumerate(train_dataloader):
            tic = perf_counter()

            # zero the parameter gradients
            optimizer.zero_grad()

            # input data
            device = next(model.parameters()).device  # model device
            data = move_data_to_device(data, device)
            train_input = (data[key] for key in model_train_keys)
            loss_input = (data[key] for key in loss_keys)

            # forward + backward + optimize
            output = model.forward_train(*train_input)
            losses = loss(output, *loss_input)
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
                log_str = (
                    f"[{epoch + 1}, {i + 1:5d} / {len(train_dataloader)}] "
                )
                for k, v in running_losses.items():
                    log_str += f"{k}: {v / log_step:.3f}, "
                print(log_str.rstrip(", "))
                running_losses = {}

        scheduler.step()
        torch.save(model.state_dict(), f"{save_prefix}_{epoch + 1}.pt")
        testing_loop(
            test_dataloader, evaluators, metric, model, model_test_keys
        )
    print("training done.")
