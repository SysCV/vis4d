"""CLI interface."""

from __future__ import annotations

from absl import app

from vis4d.common import ArgsType
from vis4d.common.logging import rank_zero_info
from vis4d.config import instantiate_classes
from vis4d.config.replicator import replicate_config
from vis4d.config.typing import ExperimentConfig

from .experiment import run_experiment
from .flag import _CKPT, _CONFIG, _GPUS, _RESUME, _SHOW_CONFIG, _SLURM, _SWEEP


def main(argv: ArgsType) -> None:
    """Main entry point for the CLI.

    Example to run this script:
    >>> python -m vis4d.engine.run --config vis4d/zoo/faster_rcnn/faster_rcnn_coco.py
    With parameter sweep config:
    >>> python -m vis4d.engine.run fit --config vis4d/zoo/faster_rcnn/faster_rcnn_coco.py --sweep vis4d/zoo/faster_rcnn/faster_rcnn_coco.py
    """
    # Get config
    assert len(argv) > 1, "Mode must be specified: `fit` or `test`"
    mode = argv[1]
    assert mode in {"fit", "test"}, f"Invalid mode: {mode}"
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
            _CKPT.value,
            _RESUME.value,
        )


def entrypoint() -> None:
    """Entry point for the CLI."""
    app.run(main)


if __name__ == "__main__":
    entrypoint()
