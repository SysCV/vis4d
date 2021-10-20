"""Base class for VisT models."""
import abc
from collections.abc import Iterable
from typing import Callable, Dict, List, Optional, Tuple, no_type_check

import pytorch_lightning as pl
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from torch.optim import Optimizer

from ..common.registry import ABCRegistryHolder
from ..struct import Boxes2D, InputSample, LossesType, ModelOutput
from .optimize import (
    BaseLRScheduler,
    BaseLRSchedulerConfig,
    BaseOptimizer,
    BaseOptimizerConfig,
    build_lr_scheduler,
    build_optimizer,
    get_warmup_lr,
)


class BaseModelConfig(PydanticBaseModel, extra="allow"):
    """Config for default VisT tracker."""

    type: str = Field(...)
    category_mapping: Optional[Dict[str, int]] = None
    image_channel_mode: str = "RGB"
    optimizer: BaseOptimizerConfig = BaseOptimizerConfig()
    lr_scheduler: BaseLRSchedulerConfig = BaseLRSchedulerConfig()


class BaseModel(pl.LightningModule, metaclass=ABCRegistryHolder):
    """Base VisT model class."""

    def __init__(self, cfg: BaseModelConfig):
        """Init."""
        super().__init__()
        self.cfg = cfg

    @abc.abstractmethod
    def forward_train(
        self,
        batch_inputs: List[List[InputSample]],
    ) -> LossesType:
        """Forward pass during training stage.

        Args:
            batch_inputs: Model input. Batched, including possible reference
            views.

        Returns:
            LossesType: A dict of scalar loss tensors.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self,
        batch_inputs: List[List[InputSample]],
    ) -> ModelOutput:
        """Forward pass during testing stage.

        Args:
            batch_inputs: Model input (batched).

        Returns:
            ModelOutput: Dict of Scalabel results (List[Label]), e.g. tracking
            and separate detection result.
        """
        raise NotImplementedError

    def configure_optimizers(
        self,
    ) -> Tuple[List[BaseOptimizer], List[BaseLRScheduler]]:
        """Configure optimizers and schedulers of model."""
        optimizer = build_optimizer(self.parameters(), self.cfg.optimizer)
        scheduler = build_lr_scheduler(optimizer, self.cfg.lr_scheduler)
        return [optimizer], [scheduler]

    @no_type_check
    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        """Optimizer step plus learning rate warmup."""
        if self.trainer.global_step < self.cfg.lr_scheduler.warmup_steps:
            for pg in optimizer.param_groups:
                pg["lr"] = get_warmup_lr(
                    self.cfg.lr_scheduler,
                    self.trainer.global_step,
                    self.cfg.optimizer.lr,
                )
        elif self.trainer.global_step == self.cfg.lr_scheduler.warmup_steps:
            for pg in optimizer.param_groups:
                pg["lr"] = self.cfg.optimizer.lr

        # update params
        optimizer.step(closure=optimizer_closure)

        # if lr_scheduler is step-based, we need to call .step(), PL calls
        # .step() only after each epoch.
        if self.cfg.lr_scheduler.mode == "step":
            lr_schedulers = self.lr_schedulers()
            if isinstance(lr_schedulers, Iterable):  # pragma: no cover
                for scheduler in lr_schedulers:
                    scheduler.step()
            else:
                lr_schedulers.step()

    def training_step(  # type: ignore # pylint: disable=arguments-differ
        self, batch: List[List[InputSample]], *args, **kwargs
    ) -> LossesType:
        """Wrap training step of LightningModule. Add overall loss."""
        losses = self.forward_train(batch)
        losses["loss"] = sum(list(losses.values()))

        losses_detached = {k: v.detach() for k, v in losses.items()}
        # tensorboard logging with prefix
        self.log_dict(
            {"train/" + k: v for k, v in losses_detached.items()},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        # progress bar logging without prefix
        self.log_dict(
            losses_detached,
            prog_bar=True,
            logger=False,
            on_step=True,
            on_epoch=False,
        )
        return losses

    def test_step(  # type: ignore # pylint: disable=arguments-differ
        self, batch: List[List[InputSample]], *args, **kwargs
    ) -> ModelOutput:
        """Wrap test step of LightningModule."""
        return self.forward_test(batch)

    def validation_step(  # type: ignore # pylint: disable=arguments-differ
        self, batch: List[List[InputSample]], *args, **kwargs
    ) -> ModelOutput:
        """Wrap validation step of LightningModule."""
        return self.forward_test(batch)

    def predict_step(
        self,
        batch: List[List[InputSample]],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> ModelOutput:
        """Forward pass during prediction stage.

        Args:
            batch: Model input (batched).
            batch_idx: batch index within dataset.
            dataloader_idx: index of dataloader if there are multiple.

        Returns:
            ModelOutput: Dict of Scalabel results (List[Label]), e.g. tracking
            and separate detection result.
        """
        return self.forward_test(batch)

    @staticmethod
    def postprocess(
        original_wh: Tuple[int, int],
        output_wh: Tuple[int, int],
        detections: Boxes2D,
    ) -> None:
        """Postprocess results."""
        scale_factor = (
            original_wh[0] / output_wh[0],
            original_wh[1] / output_wh[1],
        )
        detections.scale(scale_factor)
        detections.clip(original_wh)


def build_model(cfg: BaseModelConfig, ckpt: Optional[str] = None) -> BaseModel:
    """Build VisT model and optionally load weights from ckpt."""
    registry = ABCRegistryHolder.get_registry(BaseModel)
    if cfg.type in registry:
        if ckpt is None:
            module = registry[cfg.type](cfg)
        else:
            module = registry[cfg.type].load_from_checkpoint(ckpt, cfg=cfg)  # type: ignore # pragma: no cover # pylint: disable=line-too-long
        assert isinstance(module, BaseModel)
        return module
    raise NotImplementedError(f"Model {cfg.type} not found.")
