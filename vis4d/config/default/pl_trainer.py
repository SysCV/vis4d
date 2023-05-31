"""Default runtime configuration for PyTorch Lightning."""
import inspect

import pytorch_lightning as pl

from vis4d.config import FieldConfigDict


def get_default_pl_trainer_cfg(config: FieldConfigDict) -> FieldConfigDict:
    """Get PyTorch Lightning Trainer config."""
    pl_trainer = FieldConfigDict()

    # PL Trainer arguments
    for k, v in inspect.signature(pl.Trainer).parameters.items():
        if not k in {"callbacks", "devices", "logger", "strategy"}:
            pl_trainer[k] = v.default

    # PL Trainer settings
    pl_trainer.benchmark = config.get("benchmark")
    pl_trainer.use_distributed_sampler = False
    pl_trainer.num_sanity_val_steps = 0

    # logger
    pl_trainer.enable_progress_bar = False

    # Default Trainer arguments
    pl_trainer.work_dir = config.work_dir
    pl_trainer.exp_name = config.experiment_name
    pl_trainer.version = config.version
    pl_trainer.find_unused_parameters = False
    pl_trainer.checkpoint_period = 1
    pl_trainer.wandb = False

    return pl_trainer
