"""CLI interface."""
from __future__ import annotations

from absl import app, flags

from vis4d.common import ArgsType
from vis4d.common.logging import rank_zero_info
from vis4d.config import instantiate_classes
from vis4d.config.common.types import ExperimentConfig
from vis4d.config.replicator import replicate_config
from vis4d.engine.parser import DEFINE_config_file

from .experiment import run_experiment

# TODO: Currently this does not allow to load multpile config files.
# Would be nice to extend functionality to chain multiple config files using
# e.g. --config=model_1.py --config=loader_args.py
# or --config=my_config.py --config.train_dl=different_dl.py

_CONFIG = DEFINE_config_file("config", method_name="get_config")
_SWEEP = DEFINE_config_file("sweep", method_name="get_sweep")
_GPUS = flags.DEFINE_integer("gpus", default=0, help="Number of GPUs")
_SHOW_CONFIG = flags.DEFINE_bool(
    "print-config", default=False, help="If set, prints the configuration."
)
_SLURM = flags.DEFINE_bool(
    "slurm", default=False, help="If set, setup slurm running jobs."
)


def main(argv: ArgsType) -> None:
    """Main entry point for the CLI.

    Example to run this script:
    >>> python -m vis4d.engine.cli --config vis4d/zoo/faster_rcnn/faster_rcnn_coco.py
    With parameter sweep config:
    >>> python -m vis4d.engine.cli fit --config vis4d/zoo/faster_rcnn/faster_rcnn_coco.py --sweep  vis4d/zoo/faster_rcnn/faster_rcnn_coco.py
    """
    # Get config
    assert len(argv) > 1, "Mode must be specified: `fit` or `test`"
    mode = argv[1]
    assert mode in {"fit", "test"}, f"Invalid mode: {mode}"
    experiment_config: ExperimentConfig = _CONFIG.value

<<<<<<< HEAD
    # Setup logging
    logger_vis4d = logging.getLogger("vis4d")
    log_dir = os.path.join(config.output_dir, f"log_{config.timestamp}.txt")
    setup_logger(logger_vis4d, log_dir)

    rank_zero_info("Environment info: %s", get_pretty_env_info())

    # PyTorch Setting
    set_tf32(False)
    if "benchmark" in config:
        torch.backends.cudnn.benchmark = config.benchmark

    # TODO: Add random seed and DDP
    if _SHOW_CONFIG.value:
        rank_zero_info(pprints_config(config))

    # Instantiate classes
    model = instantiate_classes(config.model)

    if config.get("sync_batchnorm", False):
        if num_gpus > 1:
            rank_zero_info(
                "SyncBN enabled, converting BatchNorm layers to"
                " SyncBatchNorm layers."
            )
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            rank_zero_warn(
                "use_sync_bn is True, but not in a distributed setting."
                " BatchNorm layers are not converted."
            )

    # Callbacks
    callbacks = [instantiate_classes(cb) for cb in config.callbacks]

    # Setup DDP & seed
    seed = config.get("seed", init_random_seed())
    if num_gpus > 1:
        ddp_setup(slurm=_SLURM.value)

        # broadcast seed to all processes
        seed = broadcast(seed)

    # Setup Dataloaders & seed
    if mode == "fit":
        set_random_seed(seed)
        _info(f"[rank {get_rank()}] Global seed set to {seed}")
        train_dataloader = instantiate_classes(config.data.train_dataloader)
        train_data_connector = instantiate_classes(config.train_data_connector)
        optimizers = set_up_optimizers(config.optimizers, model)
        loss = instantiate_classes(config.loss)
    else:
        train_dataloader = None
        train_data_connector = None

    test_dataloader = instantiate_classes(config.data.test_dataloader)
    test_data_connector = instantiate_classes(config.test_data_connector)

    # Setup Model
    if num_gpus == 0:
        device = torch.device("cpu")
    else:
        rank = get_local_rank()
        device = torch.device(f"cuda:{rank}")

    model.to(device)

    if num_gpus > 1:
        model = DDP(  # pylint: disable=redefined-variable-type
            model, device_ids=[rank]
        )

    # Setup Callbacks
    for cb in callbacks:
        cb.setup()

    trainer = Trainer(
        device=device,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        train_data_connector=train_data_connector,
        test_data_connector=test_data_connector,
        callbacks=callbacks,
        num_epochs=config.params.get("num_epochs", -1),
        num_steps=config.params.get("num_steps", -1),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 1),
        val_check_interval=config.get("val_check_interval", None),
        use_ema_model_for_test=config.get("use_ema_model_for_test", True),
    )

    # TODO: Parameter sweep. Where to save the results? What name for the run?
=======
>>>>>>> main
    if _SWEEP.value is not None:
        # Perform parameter sweep
        rank_zero_info(
            "Found Parameter Sweep in config file. Running Parameter Sweep..."
        )
        experiment_config = _CONFIG.value
        sweep_config = instantiate_classes(_SWEEP.value)

        for run_id, config in enumerate(
            replicate_config(
                experiment_config,
                method=sweep_config.method,
                sampling_args=sweep_config.sampling_args,
                fstring=sweep_config.get("suffix", ""),
            )
        ):
            rank_zero_info(
                "Running experiment #%d: %s",
                run_id,
                config.experiment_name,
            )
            # Run single experiment
            run_experiment(
                experiment_config,
                mode,
                _GPUS.value,
                _SHOW_CONFIG.value,
                _SLURM.value,
            )

    else:
        # Run single experiment
        run_experiment(
            experiment_config,
            mode,
            _GPUS.value,
            _SHOW_CONFIG.value,
            _SLURM.value,
        )


def entrypoint() -> None:
    """Entry point for the CLI."""
    app.run(main)


if __name__ == "__main__":
    entrypoint()
