"""Base class for Vis4D models."""
import abc
from collections.abc import Iterable
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    no_type_check,
)

import pytorch_lightning as pl
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from torch.optim import Optimizer

from ..common.registry import RegistryHolder
from ..struct import DictStrAny, InputSample, LossesType, ModelOutput
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
    """Config for default Vis4D model."""

    type: str = Field(...)
    category_mapping: Optional[Dict[str, int]] = None
    image_channel_mode: str = "RGB"
    optimizer: BaseOptimizerConfig = BaseOptimizerConfig()
    lr_scheduler: BaseLRSchedulerConfig = BaseLRSchedulerConfig()
    freeze: bool = False
    freeze_parameters: Optional[List[str]] = None


class BaseModel(pl.LightningModule, metaclass=RegistryHolder):
    """Base Vis4D model class."""

    def __init__(self, cfg: BaseModelConfig):
        """Init."""
        super().__init__()
        self.cfg = cfg

    def forward(  # type: ignore[override]
        self, batch_inputs: List[InputSample]
    ) -> Union[LossesType, ModelOutput]:  # pragma: no cover
        """Forward."""
        if self.training:
            return self.forward_train(batch_inputs)
        return self.forward_test(batch_inputs)

    @abc.abstractmethod
    def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
        """Forward pass during training stage.

        Args:
            batch_inputs: List of batched model inputs. One InputSample
            contains all batch elements of a single view. One view is either
            the key frame or a reference frame.

        Returns:
            LossesType: A dict of scalar loss tensors.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
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
        self, batch: List[InputSample], *args, **kwargs
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
        self, batch: List[InputSample], *args, **kwargs
    ) -> ModelOutput:
        """Wrap test step of LightningModule."""
        return self.forward_test(batch)

    def validation_step(  # type: ignore # pylint: disable=arguments-differ
        self, batch: List[InputSample], *args, **kwargs
    ) -> ModelOutput:
        """Wrap validation step of LightningModule."""
        return self.forward_test(batch)

    def predict_step(
        self,
        batch: List[InputSample],
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

    def on_train_start(self) -> None:
        """Called at the beginning of training after sanity check."""
        self.freeze_parameters()

    def freeze_parameters(self) -> None:
        """Freeze model parameters according to config."""
        if not self.cfg.freeze:
            return
        if self.cfg.freeze_parameters is not None:
            pnames, params = [], []
            for freeze_param in self.cfg.freeze_parameters:
                for name, param in self.named_parameters():
                    if name.startswith(freeze_param) and name not in pnames:
                        params.append(param)
                        pnames.append(name)
        else:  # pragma: no cover
            params = self.parameters()
        for param in params:
            param.requires_grad = False

    def unfreeze(self) -> None:  # pragma: no cover
        """Unfreeze all parameters for training."""
        for param in self.parameters():
            param.requires_grad = True
        self.freeze_parameters()

        self.train()

    @classmethod
    def _load_model_state(  # type: ignore
        cls, checkpoint: DictStrAny, strict: bool = True, **cls_kwargs_new: Any
    ) -> "BaseModel":
        """Legacy checkpoint support in _load_model_state."""
        is_legacy = cls_kwargs_new.pop("legacy_ckpt", False)
        if is_legacy:
            rev_keys = [
                ("mm_detector.", ""),
                ("roi_head", "roi_head.mm_roi_head"),
                ("rpn_head", "rpn_head.mm_dense_head"),
                ("backbone", "backbone.mm_backbone"),
                ("neck", "backbone.neck.mm_neck"),
            ]
            new_state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                for pattern, replacement in rev_keys:
                    k = k.replace(pattern, replacement)
                new_state_dict[k] = v
            checkpoint["state_dict"] = new_state_dict
        return super()._load_model_state(  # type: ignore
            checkpoint, strict=strict, **cls_kwargs_new
        )


def build_model(
    cfg: BaseModelConfig,
    ckpt: Optional[str] = None,
    strict: bool = True,
    legacy_ckpt: bool = False,
) -> BaseModel:
    """Build Vis4D model and optionally load weights from ckpt."""
    registry = RegistryHolder.get_registry(BaseModel)
    if cfg.type in registry:
        if ckpt is None:
            module = registry[cfg.type](cfg)
        else:
            module = registry[cfg.type].load_from_checkpoint(  # type: ignore # pragma: no cover # pylint: disable=line-too-long
                ckpt, strict=strict, cfg=cfg, legacy_ckpt=legacy_ckpt
            )
        assert isinstance(module, BaseModel)
        return module
    raise NotImplementedError(f"Model {cfg.type} not found.")
