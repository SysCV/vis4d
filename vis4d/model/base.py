"""Base class for Vis4D models."""
import abc
import os.path as osp
import re
from collections import OrderedDict
from collections.abc import Iterable
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    no_type_check,
)

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cli import instantiate_class
from pytorch_lightning.utilities.rank_zero import (
    rank_zero_info,
    rank_zero_warn,
)
from torch.nn.parameter import Parameter
from torch.optim import Optimizer, lr_scheduler
from torch.utils.model_zoo import load_url
from torchmetrics import MeanMetric

from ..common.io import HDF5Backend
from ..common.registry import RegistryHolder
from ..common.utils.distributed import get_rank, get_world_size
from ..struct import (
    ArgsType,
    DictStrAny,
    InputSample,
    LossesType,
    ModelOutput,
    ModuleCfg,
)
from .optimize import BaseLRWarmup, LinearLRWarmup

try:
    from mmcv.runner.fp16_utils import wrap_fp16_model

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

DEFAULT_OPTIM = {
    "class_path": "torch.optim.SGD",
    "init_args": {
        "lr": 1.0e-3,
        "momentum": 0.9,
        "weight_decay": 0.0001,
    },
}

DEFAULT_SCHEDULER = {
    "class_path": "torch.optim.lr_scheduler.StepLR",
    "mode": "epoch",
    "init_args": {"step_size": 10},
}


class BaseModel(pl.LightningModule, metaclass=RegistryHolder):
    """Base model class."""

    def __init__(
        self,
        *args: ArgsType,
        optimizer_init: Optional[ModuleCfg] = None,
        lr_scheduler_init: Optional[ModuleCfg] = None,
        category_mapping: Optional[Dict[str, int]] = None,
        image_channel_mode: str = "RGB",
        freeze: bool = False,
        freeze_parameters: Optional[List[str]] = None,
        inference_result_path: Optional[str] = None,
        weights: Optional[str] = None,
        strict: bool = True,
        revise_keys: Optional[List[Tuple[str, str]]] = None,
        lr_warmup: Optional[BaseLRWarmup] = None,
        **kwargs: ArgsType,
    ):
        """Init."""
        if len(args) > 0:
            rank_zero_warn(f"Found unused positional arguments: {args}")
        if len(kwargs) > 0:
            rank_zero_warn(f"Found unused keyword arguments: {kwargs}")
        super().__init__()

        self.optimizer_init = (
            optimizer_init if optimizer_init is not None else DEFAULT_OPTIM
        )
        self.lr_scheduler_init = (
            lr_scheduler_init
            if lr_scheduler_init is not None
            else DEFAULT_SCHEDULER
        )
        if not self.lr_scheduler_init.get("mode", "epoch") in [
            "step",
            "epoch",
        ]:
            raise ValueError(
                "Attribute mode of LR Scheduler must be either step or epoch, "
                f"found {self.lr_scheduler_init['mode']}"
            )

        self.category_mapping = category_mapping
        self.image_channel_mode = image_channel_mode

        self._freeze = freeze
        self._freeze_parameters = freeze_parameters
        self._weights = weights
        self._strict = strict
        self._revise_keys = revise_keys
        self.lr_warmup = (
            lr_warmup if lr_warmup is not None else LinearLRWarmup(0.001, 500)
        )
        self.inference_result_path = inference_result_path
        if self.inference_result_path is not None:
            self.data_backend = HDF5Backend()

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup model according to trainer parameters, stage, etc."""
        if (
            self.trainer is not None
            and self.trainer.precision == 16
            and MMCV_INSTALLED
        ):
            wrap_fp16_model(self)  # pragma: no cover

    def __call__(
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
    ) -> Tuple[List[Optimizer], List[lr_scheduler._LRScheduler]]:
        """Configure optimizers and schedulers of model."""
        params = self.set_param_wise_optim(self.named_parameters())
        optimizer = instantiate_class(params, self.optimizer_init)
        scheduler = instantiate_class(optimizer, self.lr_scheduler_init)
        return [optimizer], [scheduler]

    def set_param_wise_optim(
        self, params: Iterator[Tuple[str, Parameter]]
    ) -> List[DictStrAny]:
        """Setting param-wise lr and weight decay."""
        paramwise_options = self.optimizer_init.pop("paramwise_options")
        if paramwise_options is None:
            new_params = [{"params": [param]} for _, param in params]
            return new_params

        base_lr = self.optimizer_init["init_args"]["lr"]
        base_wd = self.optimizer_init["init_args"]["weight_decay"]

        # get param-wise options
        bias_lr_mult = paramwise_options.get("bias_lr_mult", 1.0)
        bias_decay_mult = paramwise_options.get("bias_decay_mult", 1.0)
        norm_decay_mult = paramwise_options.get("norm_decay_mult", 1.0)
        bboxfc_lr_mult = paramwise_options.get("bboxfc_lr_mult", 1.0)

        new_params = []
        for name, param in params:
            param_group = {"params": [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                new_params.append(param_group)
                continue

            bbox_head = (
                name.find("cls") != -1
                or name.find("reg") != -1
                or name.find("dep") != -1
                or name.find("dim") != -1
                or name.find("rot") != -1
                or name.find("2dc") != -1
            )

            # For norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically  # pylint: disable=line-too-long,fixme
            if re.search(r"(bn|gn)(\d+)?.(weight|bias)", name):
                if base_wd is not None:
                    param_group["weight_decay"] = base_wd * norm_decay_mult
            # For the other layers, overwrite both lr and weight decay of bias
            elif name.endswith(".bias") and name.find("offset") == -1:
                param_group["lr"] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group["weight_decay"] = base_wd * bias_decay_mult
            # Overwrite bbox head lr
            elif bbox_head:
                param_group["lr"] = base_lr * bboxfc_lr_mult
                rank_zero_info(f"{name} with lr_multi: {bboxfc_lr_mult}")
            new_params.append(param_group)
        return new_params

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
        if self.lr_scheduler_init.get("mode", "epoch") == "step":
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

        log_dict = {}
        metric_attributes = []
        for k, v in losses.items():
            if not hasattr(self, k):
                metric = MeanMetric()
                metric.to(self.device)
                setattr(self, k, metric)

            metric = getattr(self, k)
            metric(v.detach())
            log_dict["train/" + k] = metric
            metric_attributes += [k]

        for (k, v), k_name in zip(log_dict.items(), metric_attributes):
            self.log(
                k,
                v,
                logger=True,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                metric_attribute=k_name,
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

    def freeze_parameters(
        self, parameters: Optional[List[str]] = None
    ) -> None:
        """Freeze (given) model parameters."""
        if parameters is not None:
            pnames, params = [], []
            for freeze_param in parameters:
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
        if self._freeze:
            self.freeze_parameters(self._freeze_parameters)
        self.train()

    def on_load_checkpoint(
        self, checkpoint: DictStrAny
    ) -> None:  # pragma: no cover
        """Allow for mismatched shapes when loading checkpoints."""
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        for k in checkpoint["state_dict"]:
            if k in model_state_dict:
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
