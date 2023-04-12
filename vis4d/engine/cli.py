"""CLI interface for vis4d.

Example to run this script:
>>> python -m vis4d.engine.cli --config configs/faster_rcnn/faster_rcnn_coco.py

To run a parameter sweep:
>>> python -m vis4d.engine.cli fit --config configs/faster_rcnn/faster_rcnn_coco.py --sweep configs/faster_rcnn/faster_rcnn_coco.py
"""
from __future__ import annotations

from absl import app, flags

from vis4d.common.imports import SUBMITIT_AVAILABLE
from vis4d.common.logging import rank_zero_info
from vis4d.config.replicator import replicate_config
from vis4d.config.util import ConfigDict, instantiate_classes
from vis4d.engine.experiment import run_experiment
from vis4d.engine.parser import DEFINE_config_file

if SUBMITIT_AVAILABLE:
    import submitit

# Currently this does not allow to load multpile config files.
# Would be nice to extend functionality to chain multiple config files using
# e.g. --config=model_1.py --config=loader_args.py
# or --config=my_config.py --config.train_dl=different_dl.py
_SLURM_EXECUTOR = DEFINE_config_file("slurm", method_name="get_slurm")
_CONFIG = DEFINE_config_file("config", method_name="get_config")
_SWEEP = DEFINE_config_file("sweep", method_name="get_sweep")
_GPUS = flags.DEFINE_integer("gpus", default=0, help="Number of GPUs")
_SHOW_CONFIG = flags.DEFINE_bool(
    "print-config", default=False, help="If set, prints the configuration."
)
_SLURM = flags.DEFINE_bool(
    "slurm", default=False, help="If set, setup slurm running jobs."
)


def main(argv) -> None:  # type:ignore
    """Main entry point for the CLI.

    Example to run this script:
    >>> python -m vis4d.engine.cli --config configs/faster_rcnn/faster_rcnn_coco.py

    To run a parameter sweep:
    >>> python -m vis4d.engine.cli fit --config configs/faster_rcnn/faster_rcnn_coco.py --sweep configs/faster_rcnn/faster_rcnn_coco.py
    """
    # Get config
    mode = argv[1]
    assert mode in {"fit", "test"}, f"Invalid mode: {mode}"
    experiment_config = _CONFIG.value

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
                f"Running experiment #%d: %s",  # pylint: disable=f-string-without-interpolation, line-too-long
                run_id,
                config.experiment_name,
            )
            run_single_experiment(
                experiment_config,
                mode,
                _GPUS.value,
                _SHOW_CONFIG.value,
                _SLURM.value or _SLURM_EXECUTOR.value is not None,
            )
    else:
        # Run single experiment
        run_single_experiment(
            experiment_config,
            mode,
            _GPUS.value,
            _SHOW_CONFIG.value,
            _SLURM.value or _SLURM_EXECUTOR.value is not None,
        )


def run_single_experiment(
    config: ConfigDict,
    mode: str,
    num_gpus: int = 0,
    show_config: bool = False,
    use_slurm: bool = False,
) -> None:
    """Entry point for running a single experiment.

    This function potentially submits a standalone job to the slurm scheduler.

    Args:
        config (ConfigDict): Configuration dictionary.
        mode (str): Mode to run the experiment in. Either `fit` or `test`.
        num_gpus (int): Number of GPUs to use.
        show_config (bool): If set, prints the configuration.
        use_slurm (bool): If set, setup slurm running jobs. This will set the
            required environment variables for slurm.

    Raises:
        ValueError: If `mode` is not `fit` or `test`.
        ImportError: If `submitit` is not installed but requested.
    """
    if _SLURM_EXECUTOR.value:
        if not SUBMITIT_AVAILABLE:
            raise ImportError(
                "Submitit is not installed but required to sumbit jobs on "
                "SLURM clusters. Please install it with "
                "`pip install submitit`."
            )
        rank_zero_info("Submitting job to slurm scheduler...")

        executor = submitit.AutoExecutor(folder=config.output_dir)
        executor.update_parameters(**_SLURM_EXECUTOR.value.to_dict())
        job = executor.submit(
            run_experiment, config, mode, num_gpus, show_config, use_slurm
        )
        rank_zero_info(f"Job submitted with job id {job.job_id}")
    else:
        run_experiment(config, mode, num_gpus, show_config, use_slurm)


if __name__ == "__main__":
    app.run(main)
