"""Engine Flags."""

from absl import flags

from .parser import DEFINE_config_file

# TODO: Currently this does not allow to load multpile config files.
# Would be nice to extend functionality to chain multiple config files using
# e.g. --config=model_1.py --config=loader_args.py
# or --config=my_config.py --config.train_dl=different_dl.py

# TODO: Support resume from folder and load config directly from it.
_CONFIG = DEFINE_config_file("config", method_name="get_config")
_GPUS = flags.DEFINE_integer("gpus", default=0, help="Number of GPUs")
_CKPT = flags.DEFINE_string("ckpt", default=None, help="Checkpoint path")
_RESUME = flags.DEFINE_bool("resume", default=False, help="Resume training")
_SHOW_CONFIG = flags.DEFINE_bool(
    "print-config", default=False, help="If set, prints the configuration."
)
_SWEEP = DEFINE_config_file("sweep", method_name="get_sweep")
_SLURM = flags.DEFINE_bool(
    "slurm", default=False, help="If set, setup slurm running jobs."
)
_VIS = flags.DEFINE_bool(
    "vis",
    default=False,
    help="If set, running visualization using visualizer callback.",
)


__all__ = [
    "_CONFIG",
    "_GPUS",
    "_CKPT",
    "_RESUME",
    "_SHOW_CONFIG",
    "_SWEEP",
    "_SLURM",
    "_VIS",
]
