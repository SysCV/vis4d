"""LightningModule that wraps around the models, losses and optims."""
from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from ml_collections import ConfigDict
from torch import Tensor, nn

from vis4d.common.ckpt import load_model_checkpoint
from vis4d.common.distributed import broadcast
from vis4d.common.logging import rank_zero_info
from vis4d.common.typing import DictStrAny
from vis4d.common.util import init_random_seed
from vis4d.config import instantiate_classes
from vis4d.data.typing import DictData
from vis4d.engine.connectors import DataConnector
from vis4d.engine.loss_module import LossModule
from vis4d.engine.optim import set_up_optimizers, BaseLRWarmup


class TrainingModule(pl.LightningModule):
    """LightningModule that wraps around the vis4d implementations.

    This is a wrapper around the vis4d implementations that allows to use
    pytorch-lightning for training and testing.
    """

    def __init__(
        self,
        model_cfg: ConfigDict,
        optimizers_cfg: list[ConfigDict],
        loss_module: None | LossModule,
        train_data_connector: None | DataConnector,
        test_data_connector: None | DataConnector,
        hyper_parameters: DictStrAny,
        seed: None | int = None,
        ckpt_path: None | str = None,
    ) -> None:
        """Initialize the TrainingModule.

        Args:
            model: The model config  to train.
            optimizers: The optimizers to use. Will be wrapped into a pytorch
                optimizer.
            loss_module: The loss function to use.
            train_data_connector: The data connector to use.
            test_data_connector: The data connector to use.
            data_connector: The data connector to use.
            hyper_parameters (DictStrAny): The hyper parameters to use.
            seed (int, optional): The integer value seed for global random
                state. Defaults to None.
            ckpt_path (str, optional): The path to the checkpoint to load.
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

        self.model: nn.Module
        self.lr_warmups: list[None | dict[str, BaseLRWarmup | bool]] = []

    def setup(self, stage: str) -> None:
        """Setup the model."""
        if stage == "fit":
            if self.seed is None:
                seed = init_random_seed()
                seed = broadcast(seed)
            else:
                seed = self.seed

            seed_everything(seed, workers=True)
            rank_zero_info(f"Global seed set to {seed}")

            self.hyper_parameters["seed"] = seed
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
        losses = self.loss_module(out, batch)

        metrics = {}
        if isinstance(losses, Tensor):
            total_loss = losses
        else:
            total_loss = sum(list(losses.values()))
            for k, v in losses.items():
                metrics[k] = v.detach().cpu().item()

        metrics["loss"] = total_loss.detach().cpu().item()

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
        out = self.model(**self.test_data_connector(batch))
        return out

    def configure_optimizers(self):
        """Return the optimizer to use."""
        optims = set_up_optimizers(self.optimizers_cfg, [self.model])

        optimizers = []
        for optim in optims:
            if optim.lr_scheduler is not None:
                lr_scheduler = {
                    "scheduler": optim.lr_scheduler,
                    "interval": "epoch" if optim.epoch_based_lr else "step",
                }
            else:
                lr_scheduler = None

            optimizers.append(
                {"optimizer": optim.optimizer, "lr_scheduler": lr_scheduler}
            )

            if optim.lr_warmup is not None:
                self.lr_warmups.append(
                    {
                        "warmup": optim.lr_warmup,
                        "epoch_based": optim.epoch_based_warmup,
                    }
                )
            else:
                self.lr_warmups.append(None)

        return optimizers
