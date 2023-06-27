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
    num_gpus = _GPUS.value
    experiment_config: ExperimentConfig = _CONFIG.value

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

    if num_gpus > 1:
        destroy_process_group()


if __name__ == "__main__":
    app.run(main)
