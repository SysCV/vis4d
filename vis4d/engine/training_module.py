"""LightningModule that wraps around the models, losses and optims."""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.core.optimizer import LightningOptimizer
from ml_collections import ConfigDict
from torch import nn
from torch.optim.optimizer import Optimizer

from vis4d.common.ckpt import load_model_checkpoint
from vis4d.common.distributed import broadcast
from vis4d.common.imports import FVCORE_AVAILABLE
from vis4d.common.logging import rank_zero_info
from vis4d.common.typing import DictStrAny, GenericFunc
from vis4d.common.util import init_random_seed
from vis4d.config import instantiate_classes
from vis4d.config.typing import OptimizerConfig
from vis4d.data.typing import DictData
from vis4d.engine.connectors import DataConnector
from vis4d.engine.loss_module import LossModule
from vis4d.engine.optim import LRSchedulerWrapper, set_up_optimizers
from vis4d.model.adapter.flops import IGNORED_OPS, FlopsModelAdapter

if FVCORE_AVAILABLE:
    from fvcore.nn import FlopCountAnalysis


class TrainingModule(pl.LightningModule):
    """LightningModule that wraps around the vis4d implementations.

    This is a wrapper around the vis4d implementations that allows to use
    pytorch-lightning for training and testing.
    """

    def __init__(
        self,
        model_cfg: ConfigDict,
        optimizers_cfg: list[OptimizerConfig],
        loss_module: None | LossModule,
        train_data_connector: None | DataConnector,
        test_data_connector: None | DataConnector,
        hyper_parameters: DictStrAny | None = None,
        seed: int = -1,
        ckpt_path: None | str = None,
        compute_flops: bool = False,
        check_unused_parameters: bool = False,
    ) -> None:
        """Initialize the TrainingModule.

        Args:
            model_cfg: The model config.
            optimizers_cfg: The optimizers config.
            loss_module: The loss module.
            train_data_connector: The data connector to use.
            test_data_connector: The data connector to use.
            data_connector: The data connector to use.
            hyper_parameters (DictStrAny | None, optional): The hyper
                parameters to use. Defaults to None.
            seed (int, optional): The integer value seed for global random
                state. Defaults to -1. If -1, a random seed will be generated.
            ckpt_path (str, optional): The path to the checkpoint to load.
                Defaults to None.
            compute_flops (bool, optional): If to compute the FLOPs of the
                model. Defaults to False.
            check_unused_parameters (bool, optional): If to check the
                unused parameters. Defaults to False.
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.optimizers_cfg = optimizers_cfg
        self.loss_module = loss_module
        self.train_data_connector = train_data_connector
        self.test_data_connector = test_data_connector
        self.hyper_parameters = hyper_parameters
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.compute_flops = compute_flops
        self.check_unused_parameters = check_unused_parameters

        # Create model placeholder
        self.model: nn.Module

    def setup(self, stage: str) -> None:
        """Setup the model."""
        if stage == "fit":
            if self.seed == -1:
                self.seed = init_random_seed()
                self.seed = broadcast(self.seed)
            self.trainer.seed = self.seed  # type: ignore

            seed_everything(self.seed, workers=True)
            rank_zero_info(f"Global seed set to {self.seed}")

            if self.hyper_parameters is not None:
                self.hyper_parameters["seed"] = self.seed
                if "checkpoint_callback" in self.hyper_parameters:
                    self.hyper_parameters.pop("checkpoint_callback")
                self.save_hyperparameters(self.hyper_parameters)

        # Instantiate the model after the seed has been set
        self.model = instantiate_classes(self.model_cfg)

        if self.ckpt_path is not None:
            load_model_checkpoint(
                self.model,
                self.ckpt_path,
                rev_keys=[(r"^model\.", ""), (r"^module\.", "")],
            )

    def forward(  # type: ignore # pylint: disable=arguments-differ
        self, data: DictData
    ) -> Any:
        """Forward pass through the model."""
        if self.training:
            assert self.train_data_connector is not None
            return self.model(**self.train_data_connector(data))
        assert self.test_data_connector is not None
        return self.model(**self.test_data_connector(data))

    def training_step(  # type: ignore # pylint: disable=arguments-differ,line-too-long,unused-argument
        self, batch: DictData, batch_idx: int
    ) -> Any:
        """Perform a single training step."""
        assert self.train_data_connector is not None
        out = self.model(**self.train_data_connector(batch))

        assert self.loss_module is not None
        total_loss, metrics = self.loss_module(out, batch)

        return {
            "loss": total_loss,
            "metrics": metrics,
            "predictions": out,
        }

    def validation_step(  # pylint: disable=arguments-differ,line-too-long,unused-argument
        self, batch: DictData, batch_idx: int, dataloader_idx: int = 0
    ) -> DictData:
        """Perform a single validation step."""
        assert self.test_data_connector is not None
        out = self.model(**self.test_data_connector(batch))
        return out

    def test_step(  # pylint: disable=arguments-differ,line-too-long,unused-argument
        self, batch: DictData, batch_idx: int, dataloader_idx: int = 0
    ) -> DictData:
        """Perform a single test step."""
        assert self.test_data_connector is not None

        if self.compute_flops:
            flatten_inputs = [
                self.test_data_connector(batch)[key]
                for key in self.test_data_connector(batch)
            ]

            flops_model = FlopsModelAdapter(
                self.model, self.test_data_connector
            )

            if not FVCORE_AVAILABLE:
                raise RuntimeError(
                    "Please install fvcore to compute FLOPs of the model."
                )

            flop_analyzer = FlopCountAnalysis(  # pylint: disable=possibly-used-before-assignment, line-too-long
                flops_model, flatten_inputs
            )

            flop_analyzer.set_op_handle(**{k: None for k in IGNORED_OPS})

            flops = flop_analyzer.total() / 1e9

            rank_zero_info(f"Flops: {flops:.2f} Gflops")

        out = self.model(**self.test_data_connector(batch))
        return out

    def configure_optimizers(self) -> Any:  # type: ignore
        """Return the optimizer to use."""
        self.trainer.fit_loop.setup_data()
        steps_per_epoch = len(self.trainer.train_dataloader)  # type: ignore
        return set_up_optimizers(
            self.optimizers_cfg, [self.model], steps_per_epoch
        )

    def lr_scheduler_step(  # type: ignore # pylint: disable=arguments-differ,line-too-long,unused-argument
        self, scheduler: LRSchedulerWrapper, metric: Any | None = None
    ) -> None:
        """Perform a step on the lr scheduler."""
        # TODO: Support metric if needed
        scheduler.step(self.current_epoch)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer | LightningOptimizer,
        optimizer_closure: GenericFunc | None = None,
    ) -> None:
        """Optimizer step."""
        if self.check_unused_parameters:
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    rank_zero_info(name)

        optimizer.step(closure=optimizer_closure)
