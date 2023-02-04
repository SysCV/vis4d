"""Example config for a grid search over learning rate."""
from ml_collections import ConfigDict

from vis4d.config.replicator import linspace_sampler
from vis4d.config.util import class_config


def get_config() -> ConfigDict:
    """Returns the config dict for a grid search over learning rate.

    Returns:
        ConfigDict: The configuration that can be used to run a grid search.
            It can be passed to replicate_config to create a list of configs
            that can be used to run a grid search.
    """
    config = ConfigDict()
    config.method = "grid"
    config.sampling_args = [
        [
            "engine.lr",
            class_config(
                linspace_sampler, min_value=0.001, max_value=0.01, n_steps=3
            ),
        ]
    ]

    return config
