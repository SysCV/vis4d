# pylint: disable=consider-using-alias,unsubscriptable-object
"""Base class for Vis4D models."""
from __future__ import annotations

import os.path as osp
import re
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any, Callable, List, no_type_check

import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import instantiate_class
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.utils.model_zoo import load_url
from torchmetrics import MeanMetric

from vis4d.common import DictStrAny, LossesType, ModelOutput
from vis4d.common.distributed import get_rank, get_world_size
from vis4d.common.logging import rank_zero_info
from vis4d.data.typing import DictData

from ..optim.warmup import BaseLRWarmup, LinearLRWarmup

DEFAULT_OPTIM = {
    "class_path": "torch.optim.SGD",
    "init_args": {
        "lr": 1.0e-3,
        "momentum": 0.9,
        "weight_decay": 0.0001,
    },
}


def default_data_connector(
    mode: str, data: DictData  # pylint: disable=unused-argument
) -> DictStrAny:
    """Default data connector forwards input with key data."""
    return dict(data=data)


class DefaultOptimizer(
    pl.LightningModule
):  # pylint: disable=too-many-ancestors
    """Default optimization routine."""

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        data_connector: Callable[
            [str, DictData], DictStrAny
        ] = default_data_connector,
        optimizer_init: DictStrAny | None = None,
        lr_scheduler_init: DictStrAny | None = None,
        freeze: bool = False,
        freeze_parameters: List[str] | None = None,
        weights: str | None = None,
        strict: bool = True,
        revise_keys: list[tuple[str, str]] | None = None,
        lr_warmup: BaseLRWarmup | None = None,
    ):
        """Creates an instance of the class."""
        super().__init__()
        self.optimizer_init = (
            optimizer_init if optimizer_init is not None else DEFAULT_OPTIM
        )
        self.lr_scheduler_init = lr_scheduler_init
        if (
            self.lr_scheduler_init is not None
            and not self.lr_scheduler_init.get("mode", "epoch")
            in set(("step", "epoch"))
        ):
            raise ValueError(
                "Attribute mode of LR Scheduler must be either step or epoch, "
                f"found {self.lr_scheduler_init['mode']}"
            )
        self.model = model
        self.model_loss = loss
        self.data_connector = data_connector

        self._freeze = freeze
        self._freeze_parameters = freeze_parameters
        self._weights = weights
        self._strict = strict
        self._revise_keys = revise_keys
        self.lr_warmup = (
            lr_warmup if lr_warmup is not None else LinearLRWarmup(0.001, 500)
        )

    def configure_optimizers(
        self,
    ) -> tuple[list[Optimizer], list[lr_scheduler._LRScheduler]]:
        """Configure optimizers and schedulers of model."""
        optimizer = instantiate_class(self.parameters(), self.optimizer_init)
        if self.lr_scheduler_init is not None:
            scheduler = instantiate_class(optimizer, self.lr_scheduler_init)
            return [optimizer], [scheduler]
        return [optimizer]

    @no_type_check
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer | pl.core.LightningOptimizer,
        optimizer_idx: int = 0,
        optimizer_closure: Callable[[], Any] | None = None,
        on_tpu: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        """Optimizer step plus learning rate warmup."""
        base_lr = optimizer.defaults.get("lr", None)
        if base_lr is None:
            raise ValueError(
                "Couldn't determine base LR from optimizer defaults: "
                f"{optimizer.defaults}"
            )

        if self.trainer.global_step < self.lr_warmup.warmup_steps:
            for pg in optimizer.param_groups:
                pg["lr"] = self.lr_warmup(self.trainer.global_step, base_lr)
        elif self.trainer.global_step == self.lr_warmup.warmup_steps:
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr

        # update params
        optimizer.step(closure=optimizer_closure)

        # if lr_scheduler is step-based, we need to call .step(), PL calls
        # .step() only after each epoch.
        if (
            self.lr_scheduler_init is not None
            and self.lr_scheduler_init.get("mode", "epoch") == "step"
        ):
            lr_schedulers = self.lr_schedulers()
            if isinstance(lr_schedulers, Iterable):  # pragma: no cover
                for scheduler in lr_schedulers:
                    scheduler.step()
            else:
                lr_schedulers.step()

    def _log_metric(
        self, key: str, value: torch.Tensor, prefix: str = ""
    ) -> None:
        """Log a scalar tensor metric with a certain key."""
        if not hasattr(self, key):
            metric = MeanMetric()
            metric.to(self.device)
            setattr(self, key, metric)

        metric = getattr(self, key)
        metric(value.detach())
        self.log(
            prefix + key,
            metric,
            logger=True,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            metric_attribute=key,
        )

    def training_step(  # type: ignore # pylint: disable=arguments-differ
        self, batch: DictData, *args, **kwargs
    ) -> LossesType:
        """Wrap training step of LightningModule. Add overall loss."""
        train_input = self.data_connector("train", batch)
        loss_input = self.data_connector("loss", batch)

        # forward + backward + optimize
        output = self.model(**train_input)
        losses = self.model_loss(output, **loss_input)
        losses["loss"] = sum(list(losses.values()))

        for k, v in losses.items():
            self._log_metric(k, v, prefix="train/")
        return losses

    def test_step(  # type: ignore # pylint: disable=arguments-differ
        self, batch: DictData, *args, **kwargs
    ) -> ModelOutput:
        """Wrap test step of LightningModule."""
        test_input = self.data_connector("test", batch)
        return self.model(**test_input)

    def validation_step(  # type: ignore # pylint: disable=arguments-differ
        self, batch: DictData, *args, **kwargs
    ) -> ModelOutput:
        """Wrap validation step of LightningModule."""
        return self.test_step(batch, *args, **kwargs)

    def predict_step(  # type: ignore # pylint: disable=arguments-differ
        self, batch: DictData, *args, **kwargs
    ) -> ModelOutput:
        """Forward pass during prediction stage."""
        return self.test_step(batch, *args, **kwargs)

    def on_fit_start(self) -> None:
        """Called at the beginning of fit."""
        if self._weights is not None:
            self.load_pretrained_weights(self._weights, self._strict)
        if self._freeze:
            self.freeze_parameters(self._freeze_parameters)

    def load_pretrained_weights(
        self, weights: str, strict: bool = True
    ) -> None:
        """Load pretrained weights from file / URL.

        Note: Only for training phase.
        """
        map_location = self.device
        if osp.isfile(weights):  # pragma: no cover
            filename = osp.expanduser(weights)
            checkpoint = torch.load(filename, map_location=map_location)
        elif weights.startswith("http"):
            rank, world_size = get_rank(), get_world_size()
            if rank == 0:
                checkpoint = load_url(weights, map_location=map_location)
            if world_size > 1:  # pragma: no cover
                torch.distributed.barrier()
                if rank > 0:
                    checkpoint = load_url(weights, map_location=map_location)
        else:
            raise FileNotFoundError(f"{weights} can not be found.")

        # get state_dict from checkpoint
        state_dict = (
            checkpoint["state_dict"]
            if "state_dict" in checkpoint
            else checkpoint
        )

        # strip prefix of state_dict
        if self._revise_keys is not None:
            for p, r in self._revise_keys:
                state_dict = OrderedDict(
                    {re.sub(p, r, k): v for k, v in state_dict.items()}
                )

        self.on_load_checkpoint({"state_dict": state_dict})

        # load state_dict
        self.load_state_dict(state_dict, strict=strict)

    def freeze_parameters(self, parameters: list[str] | None = None) -> None:
        """Freeze (given) model parameters."""
        if parameters is not None:
            pnames, params = [], []
            for freeze_param in parameters:
                for name, param in self.named_parameters():
                    if name.startswith(freeze_param) and name not in pnames:
                        params.append(param)
                        pnames.append(name)
            for name, module in self.named_modules():
                if name in parameters:
                    module.eval()
        else:  # pragma: no cover
            params = self.parameters()

        for param in params:
            param.requires_grad = False

    def unfreeze(self) -> None:  # pragma: no cover
        """Unfreeze all parameters for training."""
        for param in self.parameters():
            param.requires_grad = True
        if self._freeze:
            self.freeze_parameters(self._freeze_parameters)
        self.train()

    def on_load_checkpoint(
        self, checkpoint: DictStrAny
    ) -> None:  # pragma: no cover
        """Allow for mismatched shapes when loading checkpoints."""
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        for k in model_state_dict.keys():
            if k in checkpoint["state_dict"]:
                if (
                    checkpoint["state_dict"][k].shape
                    != model_state_dict[k].shape
                ):
                    rank_zero_info(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
            else:
                rank_zero_info(
                    f"Skip parameter: {k}, which is not in the checkpoint."
                )
                state_dict[k] = model_state_dict[k]
