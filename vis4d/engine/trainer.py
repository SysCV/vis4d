"""DefaultTrainer for Vis4D."""
import os.path as osp
from itertools import product
from typing import Dict, List, Optional, Tuple

import pandas
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin, DDPSpawnPlugin
from pytorch_lightning.utilities.device_parser import parse_gpu_ids
from pytorch_lightning.utilities.distributed import (
    rank_zero_info,
    rank_zero_warn,
)
from torch.utils import data

from vis4d.data.samplers import build_data_sampler

from ..common.module import build_module
from ..config import Config, Launch, default_argument_parser, parse_config
from ..data.build import Vis4DDataModule
from ..data.dataset import ScalabelDataset
from ..data.datasets import BaseDatasetLoader, Custom
from ..model import BaseModel, build_model
from ..struct import DictStrAny, ModuleCfg
from ..vis import ScalabelWriterCallback
from .evaluator import StandardEvaluatorCallback
from .utils import (
    Vis4DProgressBar,
    Vis4DTQDMProgressBar,
    setup_logging,
    split_args,
)


def default_setup(
    cfg: Config,
    trainer_args: Optional[DictStrAny] = None,
    training: bool = True,
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
            if (
                cfg.data is not None
                and "train_sampler" in cfg.data
                and cfg.data["train_sampler"] is not None
                and training
            ):
                # using custom sampler
                trainer_args["replace_sampler_ddp"] = False

    # create trainer
    trainer_args["callbacks"] = [lr_monitor, progress_bar, checkpoint]
    trainer = pl.Trainer(**trainer_args)

    # setup cmd line logging, print and save info about trainer / cfg / env
    setup_logging(output_dir, trainer_args, cfg)
    return trainer


def setup_category_mapping(
    data_cfgs: List[ModuleCfg],
    model_category_mapping: Optional[Dict[str, int]],
) -> None:
    """Setup category_mapping for each dataset."""
    for data_cfg in data_cfgs:
        if (
            "category_mapping" not in data_cfg
            or data_cfg["category_mapping"] is None
        ):
            if model_category_mapping is not None:
                # default to using model category_mapping, if exists
                data_cfg["category_mapping"] = model_category_mapping
            continue


def build_datasets(
    dset_cfgs: List[ModuleCfg], image_channel_mode: str, training: bool = True
) -> List[ScalabelDataset]:
    """Build datasets based on configs."""
    datasets: List[ScalabelDataset] = []
    for dl_cfg in dset_cfgs:
        mapper_cfg = dl_cfg.pop("sample_mapper", {})
        ref_cfg = dl_cfg.pop("ref_sampler", {})
        datasets.append(
            ScalabelDataset(
                build_module(dl_cfg, bound=BaseDatasetLoader),
                mapper_cfg,
                ref_cfg,
                training,
                image_channel_mode,
            )
        )
    return datasets


def build_callbacks(
    datasets: List[ScalabelDataset],
    out_dir: Optional[str] = None,
    is_predict: bool = False,
    visualize: bool = False,
) -> List[Callback]:
    """Build callbacks."""
    callbacks: List[Callback] = []
    for i, d in enumerate(datasets):
        out = (
            osp.join(out_dir, d.dataset.name) if out_dir is not None else None
        )
        if not is_predict:
            callbacks.append(StandardEvaluatorCallback(i, d.dataset, out))
        else:
            assert out is not None
            callbacks.append(ScalabelWriterCallback(i, out, visualize))
    return callbacks


def setup_experiment(
    cfg: Config, trainer_args: DictStrAny
) -> Tuple[pl.Trainer, BaseModel, pl.LightningDataModule]:
    """Build trainer, model, and data module."""
    # setup trainer
    is_train = cfg.launch.action == "train"
    trainer = default_setup(cfg, trainer_args, training=is_train)

    # setup model
    model = build_model(
        cfg.model,
        cfg.launch.weights if not cfg.launch.resume or not is_train else None,
        not cfg.launch.not_strict,
        cfg.launch.legacy_ckpt,
    )

    # setup category_mappings
    setup_category_mapping(cfg.train + cfg.test, cfg.model["category_mapping"])

    # build datasets
    cmode = (
        cfg.model["image_channel_mode"]
        if "image_channel_mode" in cfg.model
        else "RGB"
    )
    train_datasets = build_datasets(cfg.train, cmode) if is_train else None
    test_datasets, predict_datasets = None, None
    train_sampler: Optional[data.Sampler[List[int]]] = None
    if cfg.launch.action == "train":
        if cfg.data is not None and "train_sampler" in cfg.data:
            # build custom train sampler
            train_sampler = build_data_sampler(
                cfg.data["train_sampler"],
                data.ConcatDataset(train_datasets),
                cfg.launch.samples_per_gpu,
            )
    if cfg.launch.action == "predict":
        if cfg.launch.input_dir:
            input_dir = cfg.launch.input_dir
            if input_dir[-1] == "/":
                input_dir = input_dir[:-1]
            dataset_name = osp.basename(input_dir)
            predict_loaders = [Custom(name=dataset_name, data_root=input_dir)]
            predict_datasets = [
                ScalabelDataset(dl, {}, {}, False, cmode)
                for dl in predict_loaders
            ]
        else:
            predict_datasets = build_datasets(cfg.test, cmode, False)
    else:
        test_datasets = build_datasets(cfg.test, cmode, False)

    # build data module
    data_module = Vis4DDataModule(
        cfg.launch.samples_per_gpu,
        cfg.launch.workers_per_gpu,
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        predict_datasets=predict_datasets,
        seed=cfg.launch.seed,
        train_sampler=train_sampler,
    )

    # setup callbacks
    test_dir = osp.join(
        cfg.launch.work_dir, cfg.launch.exp_name, cfg.launch.version
    )
    if cfg.launch.action == "train":
        if test_datasets is not None and len(test_datasets) > 0:
            trainer.callbacks += build_callbacks(  # pylint: disable=no-member
                test_datasets
            )
    elif cfg.launch.action == "test":
        assert test_datasets is not None and len(
            test_datasets
        ), "No test datasets specified!"
        trainer.callbacks += build_callbacks(  # pylint: disable=no-member
            test_datasets, test_dir
        )
    elif cfg.launch.action == "predict":
        assert (
            predict_datasets is not None and len(predict_datasets) > 0
        ), "No predict datasets specified!"
        trainer.callbacks += build_callbacks(  # pylint: disable=no-member
            predict_datasets, test_dir, True, cfg.launch.visualize
        )
    elif cfg.launch.action == "tune":
        assert test_datasets is not None and len(
            test_datasets
        ), "No test datasets specified!"
        trainer.callbacks += build_callbacks(  # pylint: disable=no-member
            test_datasets, test_dir
        )
    else:
        raise NotImplementedError(f"Action {cfg.launch.action} not known!")

    return trainer, model, data_module


def train(
    trainer: pl.Trainer, model: BaseModel, data_module: pl.LightningDataModule
) -> None:
    """Training function."""
    trainer.fit(model, data_module)


def test(
    trainer: pl.Trainer, model: BaseModel, data_module: pl.LightningDataModule
) -> None:
    """Test function."""
    trainer.test(model, data_module, verbose=False)


def predict(
    trainer: pl.Trainer, model: BaseModel, data_module: pl.LightningDataModule
) -> None:
    """Prediction function."""
    trainer.predict(model, data_module)


def tune(
    trainer: pl.Trainer,
    model: BaseModel,
    data_module: pl.LightningDataModule,
    launch: Launch,
) -> None:
    """Tune function."""
    if launch.tuner_params is None:
        raise ValueError(
            "Tuner parameters not defined! Please specify "
            "tuner_params in Launch config."
        )
    if launch.tuner_metrics is None:
        raise ValueError(
            "Tuner metrics not defined! Please specify "
            "tuner_metrics in Launch config."
        )

    rank_zero_info("Starting hyperparameter search...")
    search_params = launch.tuner_params
    search_metrics = launch.tuner_metrics
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

    # setup experiment
    trainer, model, data_module = setup_experiment(cfg, trainer_args)

    if args.action == "train":
        train(trainer, model, data_module)
    elif args.action == "test":
        test(trainer, model, data_module)
    elif args.action == "predict":
        predict(trainer, model, data_module)
    elif args.action == "tune":
        tune(trainer, model, data_module, cfg.launch)
    else:
        raise NotImplementedError(f"Action {args.action} not known!")


if __name__ == "__main__":  # pragma: no cover
    cli_main()
