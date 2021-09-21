"""DefaultTrainer for VisT."""
import os.path as osp
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin, DDPSpawnPlugin, DDP2Plugin
from pytorch_lightning.utilities.device_parser import parse_gpu_ids

from ..config import Config, default_argument_parser, parse_config
from ..data import VisTDataModule, build_dataset_loaders
from ..model import build_model
from ..struct import DictStrAny
from ..vis import ScalabelWriterCallback
from .evaluator import ScalabelEvaluatorCallback
from .utils import VisTProgressBar, split_args, setup_logging


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
    if cfg.launch.seed is not None:
        pl.seed_everything(cfg.launch.seed, workers=True)

    # prepare trainer args
    if trainer_args is None:
        trainer_args = {}  # pragma: no cover
    if "trainer" in cfg.dict().keys():
        trainer_args.update(cfg.dict()["trainer"])

    # setup experiment logging
    exp_logger = pl.loggers.TensorBoardLogger(
        save_dir=cfg.launch.work_dir,
        name=cfg.launch.exp_name,
        version=cfg.launch.version,
        default_hp_metric=False,
    )
    trainer_args["logger"] = exp_logger

    # add learning rate / GPU stats monitor (logs to tensorboard)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # add progress bar (train progress separate from validation)
    progress_bar = VisTProgressBar()

    # add Model checkpointer
    checkpoint = pl.callbacks.ModelCheckpoint(
        verbose=True,
        save_last=True,
        every_n_epochs=cfg.launch.checkpoint_period,
        save_on_train_epoch_end=True,
    )

    # resume from checkpoint if specified
    output_dir = osp.join(
        cfg.launch.work_dir, cfg.launch.exp_name, cfg.launch.version
    )
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
    gpu_ids = parse_gpu_ids(trainer_args["gpus"])
    num_gpus = len(gpu_ids) if gpu_ids is not None else 0
    if num_gpus > 1:
        if trainer_args["accelerator"] == "ddp" or trainer_args["accelerator"] is None:
            ddp_plugin = DDPPlugin(find_unused_parameters=cfg.launch.find_unused_parameters)
            trainer_args["plugins"] = [ddp_plugin]
        elif trainer_args["accelerator"] == "ddp_spawn":
            ddp_plugin = DDPSpawnPlugin(find_unused_parameters=cfg.launch.find_unused_parameters)
            trainer_args["plugins"] = [ddp_plugin]
        elif trainer_args["accelerator"] == "ddp2":
            ddp_plugin = DDP2Plugin(find_unused_parameters=cfg.launch.find_unused_parameters)
            trainer_args["plugins"] = [ddp_plugin]

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
        cfg.model, cfg.launch.weights if not cfg.launch.resume else None
    )

    train_loaders, test_loaders, predict_loaders = build_dataset_loaders(
        cfg.train, cfg.test
    )
    data_module = VisTDataModule(
        cfg.launch.samples_per_gpu,
        cfg.launch.workers_per_gpu,
        train_loaders,
        test_loaders,
        predict_loaders,
        cfg.model.category_mapping,
        cfg.model.image_channel_mode,
        seed=cfg.launch.seed,
    )

    if len(test_loaders) > 0:
        assert (
            cfg.model.category_mapping is not None
        ), "Need category mapping to evaluate model!"
        evaluators = [
            ScalabelEvaluatorCallback(
                dl,
                cfg.model.category_mapping,
            )
            for dl in test_loaders
        ]
        trainer.callbacks += evaluators  # pylint: disable=no-member
    trainer.fit(model, data_module)


def test(cfg: Config, trainer_args: Optional[DictStrAny] = None) -> None:
    """Test function."""
    trainer = default_setup(cfg, trainer_args)
    model = build_model(cfg.model, cfg.launch.weights)

    train_loaders, test_loaders, predict_loaders = build_dataset_loaders(
        [], cfg.test
    )
    data_module = VisTDataModule(
        cfg.launch.samples_per_gpu,
        cfg.launch.workers_per_gpu,
        train_loaders,
        test_loaders,
        predict_loaders,
        cfg.model.category_mapping,
        cfg.model.image_channel_mode,
        seed=cfg.launch.seed,
    )

    assert len(test_loaders), "No test datasets specified!"
    assert (
        cfg.model.category_mapping is not None
    ), "Need category mapping to evaluate model!"
    out_dir = osp.join(
        cfg.launch.work_dir, cfg.launch.exp_name, cfg.launch.version
    )
    evaluators = [
        ScalabelEvaluatorCallback(
            dl, cfg.model.category_mapping, osp.join(out_dir, dl.cfg.name)
        )
        for dl in test_loaders
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
    model = build_model(cfg.model, cfg.launch.weights)

    train_loaders, test_loaders, predict_loaders = build_dataset_loaders(
        [],
        cfg.test if cfg.launch.input_dir is None else [],
        cfg.launch.input_dir,
    )
    data_module = VisTDataModule(
        cfg.launch.samples_per_gpu,
        cfg.launch.workers_per_gpu,
        train_loaders,
        test_loaders,
        predict_loaders,
        cfg.model.category_mapping,
        cfg.model.image_channel_mode,
        seed=cfg.launch.seed,
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
            osp.join(out_dir, dl.cfg.name),
            cfg.model.category_mapping,
            cfg.launch.visualize,
        )
        for dl in dataloaders
    ]
    trainer.callbacks += evaluators  # pylint: disable=no-member
    trainer.predict(model, data_module)


def cli_main() -> None:  # pragma: no cover
    """Main function when called from command line."""
    parser = default_argument_parser()
    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    vist_args, trainer_args = split_args(args)
    cfg = parse_config(vist_args)

    if args.action == "train":
        train(cfg, vars(trainer_args))
    elif args.action == "test":
        test(cfg, vars(trainer_args))
    elif args.action == "predict":
        predict(cfg, vars(trainer_args))
    else:
        raise NotImplementedError(f"Action {args.action} not known!")


if __name__ == "__main__":  # pragma: no cover
    cli_main()
