"""DefaultTrainer for Vis4D."""
import os.path as osp
from itertools import product
from typing import Optional

import pandas
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin, DDPSpawnPlugin
from pytorch_lightning.utilities.device_parser import parse_gpu_ids
from pytorch_lightning.utilities.distributed import (
    rank_zero_info,
    rank_zero_warn,
)

from ..config import Config, default_argument_parser, parse_config
from ..data import (
    build_category_mappings,
    build_data_module,
    build_dataset_loaders,
)
from ..model import build_model
from ..struct import DictStrAny
from ..vis import ScalabelWriterCallback
from .evaluator import StandardEvaluatorCallback
from .utils import (
    Vis4DProgressBar,
    Vis4DTQDMProgressBar,
    setup_logging,
    split_args,
)


def default_setup(
    cfg: Config, trainer_args: Optional[DictStrAny] = None
) -> pl.Trainer:
    """Perform some basic common setups at the beginning of a job.

    1. Set all seeds
    2. Setup callback: tensorboard logger, LRMonitor, GPUMonitor, Checkpoint
    3. Init pytorch lightning Trainer
    4. Set up cmd line logger
    5. Log basic information about environment, trainer arguments, and config
    6. Backup the args / config to the output directory
    """
    # set seeds
    pl.seed_everything(cfg.launch.seed, workers=True)

    # prepare trainer args
    if trainer_args is None:
        trainer_args = {}  # pragma: no cover
    if "trainer" in cfg.dict().keys():
        trainer_args.update(cfg.dict()["trainer"])

    # setup experiment logging
    if "logger" not in trainer_args or (
        isinstance(trainer_args["logger"], bool) and trainer_args["logger"]
    ):
        if cfg.launch.wandb:  # pragma: no cover
            exp_logger = pl.loggers.WandbLogger(
                save_dir=cfg.launch.work_dir,
                project=cfg.launch.exp_name,
                name=cfg.launch.version,
            )
        else:
            exp_logger = pl.loggers.TensorBoardLogger(  # type: ignore
                save_dir=cfg.launch.work_dir,
                name=cfg.launch.exp_name,
                version=cfg.launch.version,
                default_hp_metric=False,
            )
        trainer_args["logger"] = exp_logger

    # add learning rate / GPU stats monitor (logs to tensorboard)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # add progress bar (train progress separate from validation)
    if cfg.launch.tqdm:
        progress_bar = Vis4DTQDMProgressBar()
    else:
        progress_bar = Vis4DProgressBar()

    # add Model checkpointer
    output_dir = osp.join(
        cfg.launch.work_dir, cfg.launch.exp_name, cfg.launch.version
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=osp.join(output_dir, "checkpoints"),
        verbose=True,
        save_last=True,
        every_n_epochs=cfg.launch.checkpoint_period,
        save_on_train_epoch_end=True,
    )

    # resume from checkpoint if specified
    if cfg.launch.resume:  # pragma: no cover
        if cfg.launch.weights is not None:
            resume_path = cfg.launch.weights
        elif osp.exists(osp.join(output_dir, "checkpoints/last.ckpt")):
            resume_path = osp.join(output_dir, "checkpoints/last.ckpt")
        else:
            raise ValueError(
                "cfg.launch.resume set to True but there is no checkpoint to "
                "resume! Please specify a checkpoint via cfg.launch.weights "
                "or configure a directory that contains a checkpoint at "
                "work_dir/exp_name/version/checkpoints/last.ckpt."
            )

        trainer_args["resume_from_checkpoint"] = resume_path

    # add distributed plugin
    if "gpus" in trainer_args:  # pragma: no cover
        gpu_ids = parse_gpu_ids(trainer_args["gpus"])
        num_gpus = len(gpu_ids) if gpu_ids is not None else 0
        if num_gpus > 1:
            if (
                trainer_args["accelerator"] == "ddp"
                or trainer_args["accelerator"] is None
            ):
                ddp_plugin = DDPPlugin(
                    find_unused_parameters=cfg.launch.find_unused_parameters
                )
                trainer_args["plugins"] = [ddp_plugin]
            elif trainer_args["accelerator"] == "ddp_spawn":
                ddp_plugin = DDPSpawnPlugin(
                    find_unused_parameters=cfg.launch.find_unused_parameters
                )  # type: ignore
                trainer_args["plugins"] = [ddp_plugin]
            elif trainer_args["accelerator"] == "ddp2":
                ddp_plugin = DDP2Plugin(
                    find_unused_parameters=cfg.launch.find_unused_parameters
                )
                trainer_args["plugins"] = [ddp_plugin]
            if cfg.data.train_sampler is not None:
                # using custom sampler
                trainer_args["replace_sampler_ddp"] = False

    # create trainer
    trainer_args["callbacks"] = [lr_monitor, progress_bar, checkpoint]
    trainer = pl.Trainer(**trainer_args)

    # setup cmd line logging, print and save info about trainer / cfg / env
    setup_logging(output_dir, trainer_args, cfg)
    return trainer


def train(cfg: Config, trainer_args: Optional[DictStrAny] = None) -> None:
    """Training function."""
    trainer = default_setup(cfg, trainer_args)
    model = build_model(
        cfg.model,
        cfg.launch.weights if not cfg.launch.resume else None,
        not cfg.launch.not_strict,
        cfg.launch.legacy_ckpt,
    )

    # build dataloaders
    train_loaders, test_loaders, predict_loaders = build_dataset_loaders(
        cfg.train, cfg.test
    )

    # determine category mapping
    category_mapping = build_category_mappings(
        cfg.data, cfg.model.category_mapping
    )

    data_module = build_data_module(
        cfg.launch.samples_per_gpu,
        cfg.launch.workers_per_gpu,
        train_loaders,
        test_loaders,
        predict_loaders,
        category_mapping,
        cfg.model.image_channel_mode,
        cfg.launch.seed,
        cfg.data,
    )

    if len(test_loaders) > 0:
        evaluators = [
            StandardEvaluatorCallback(i, dl)
            for i, dl in enumerate(test_loaders)
        ]
        trainer.callbacks += evaluators  # pylint: disable=no-member
    trainer.fit(model, data_module)


def test(cfg: Config, trainer_args: Optional[DictStrAny] = None) -> None:
    """Test function."""
    trainer = default_setup(cfg, trainer_args)
    model = build_model(
        cfg.model,
        cfg.launch.weights,
        not cfg.launch.not_strict,
        cfg.launch.legacy_ckpt,
    )

    # build dataloaders
    train_loaders, test_loaders, predict_loaders = build_dataset_loaders(
        [], cfg.test
    )

    # determine category mapping
    category_mapping = build_category_mappings(
        cfg.data, cfg.model.category_mapping
    )

    data_module = build_data_module(
        cfg.launch.samples_per_gpu,
        cfg.launch.workers_per_gpu,
        train_loaders,
        test_loaders,
        predict_loaders,
        category_mapping,
        cfg.model.image_channel_mode,
        cfg.launch.seed,
        cfg.data,
    )

    assert len(test_loaders), "No test datasets specified!"
    out_dir = osp.join(
        cfg.launch.work_dir, cfg.launch.exp_name, cfg.launch.version
    )
    evaluators = [
        StandardEvaluatorCallback(i, dl, osp.join(out_dir, dl.cfg.name))
        for i, dl in enumerate(test_loaders)
    ]
    trainer.callbacks += evaluators  # pylint: disable=no-member
    trainer.test(
        model,
        data_module,
        verbose=False,
    )


def predict(cfg: Config, trainer_args: Optional[DictStrAny] = None) -> None:
    """Prediction function."""
    trainer = default_setup(cfg, trainer_args)
    model = build_model(
        cfg.model,
        cfg.launch.weights,
        not cfg.launch.not_strict,
        cfg.launch.legacy_ckpt,
    )

    # build dataloaders
    train_loaders, test_loaders, predict_loaders = build_dataset_loaders(
        [],
        cfg.test if cfg.launch.input_dir is None else [],
        cfg.launch.input_dir,
    )

    # determine category mapping
    category_mapping = build_category_mappings(
        cfg.data, cfg.model.category_mapping
    )

    data_module = build_data_module(
        cfg.launch.samples_per_gpu,
        cfg.launch.workers_per_gpu,
        train_loaders,
        test_loaders,
        predict_loaders,
        category_mapping,
        cfg.model.image_channel_mode,
        cfg.launch.seed,
        cfg.data,
    )

    out_dir = osp.join(
        cfg.launch.work_dir, cfg.launch.exp_name, cfg.launch.version
    )

    if len(predict_loaders) > 0:
        dataloaders = predict_loaders
    else:
        dataloaders = test_loaders

    assert len(dataloaders) > 0, "No datasets for prediction specified!"
    evaluators = [
        ScalabelWriterCallback(
            i, osp.join(out_dir, dl.cfg.name), cfg.launch.visualize
        )
        for i, dl in enumerate(dataloaders)
    ]
    trainer.callbacks += evaluators  # pylint: disable=no-member
    trainer.predict(model, data_module)


def tune(cfg: Config, trainer_args: Optional[DictStrAny] = None) -> None:
    """Tune function."""
    trainer = default_setup(cfg, trainer_args)
    model = build_model(
        cfg.model,
        cfg.launch.weights,
        not cfg.launch.not_strict,
        cfg.launch.legacy_ckpt,
    )

    # build dataloaders
    train_loaders, test_loaders, predict_loaders = build_dataset_loaders(
        [], cfg.test
    )

    # determine category mapping
    category_mapping = build_category_mappings(
        cfg.data, cfg.model.category_mapping
    )

    data_module = build_data_module(
        cfg.launch.samples_per_gpu,
        cfg.launch.workers_per_gpu,
        train_loaders,
        test_loaders,
        predict_loaders,
        category_mapping,
        cfg.model.image_channel_mode,
        cfg.launch.seed,
        cfg.data,
    )

    assert len(test_loaders), "No test datasets specified!"
    out_dir = osp.join(
        cfg.launch.work_dir, cfg.launch.exp_name, cfg.launch.version
    )
    evaluators = [
        StandardEvaluatorCallback(i, dl, osp.join(out_dir, dl.cfg.name))
        for i, dl in enumerate(test_loaders)
    ]
    trainer.callbacks += evaluators  # pylint: disable=no-member

    if cfg.launch.tuner_params is None:
        raise ValueError(
            "Tuner parameters not defined! Please specify "
            "tuner_params in Launch config."
        )
    if cfg.launch.tuner_metrics is None:
        raise ValueError(
            "Tuner metrics not defined! Please specify "
            "tuner_metrics in Launch config."
        )

    rank_zero_info("Starting hyperparameter search...")
    search_params = cfg.launch.tuner_params
    search_metrics = cfg.launch.tuner_metrics
    param_names = list(search_params.keys())
    param_groups = list(product(*search_params.values()))
    metrics_all = {}
    for param_group in param_groups:
        for key, value in zip(param_names, param_group):
            obj = model
            for name in key.split(".")[:-1]:
                obj = getattr(obj, name, None)  # type: ignore
                if obj is None:
                    raise ValueError(f"Attribute {name} not found in {key}!")
            setattr(obj, key.split(".")[-1], value)

        metrics = trainer.test(
            model,
            data_module,
            verbose=False,
        )
        if len(metrics) > 0:
            rank_zero_warn(
                "More than one dataloader found, but tuning "
                "requires a single dataset to tune parameters on!"
            )
        metrics_all[str(param_group)] = {
            k: v for k, v in metrics[0].items() if k in search_metrics
        }
    rank_zero_info("Done!")
    rank_zero_info("The following parameters have been considered:")
    rank_zero_info("\n" + str(list(search_params.keys())))
    rank_zero_info("Hyperparameter search result:")
    rank_zero_info("\n" + str(pandas.DataFrame.from_dict(metrics_all)))


def cli_main() -> None:  # pragma: no cover
    """Main function when called from command line."""
    parser = default_argument_parser()
    args = parser.parse_args()
    vis4d_args, trainer_args = split_args(args)
    cfg = parse_config(vis4d_args)

    if args.action == "train":
        train(cfg, trainer_args)
    elif args.action == "test":
        test(cfg, trainer_args)
    elif args.action == "predict":
        predict(cfg, trainer_args)
    elif args.action == "tune":
        tune(cfg, trainer_args)
    else:
        raise NotImplementedError(f"Action {args.action} not known!")


if __name__ == "__main__":  # pragma: no cover
    cli_main()
