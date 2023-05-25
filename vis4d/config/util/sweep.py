"""Helper functions for creating sweep configurations."""
from __future__ import annotations

from vis4d.config import ConfigDict, class_config
from vis4d.config.replicator import linspace_sampler


def linear_grid_search(
    param_names: list[str] | str,
    min_values: list[float] | float,
    max_values: list[float] | float,
    n_steps: list[int] | int,
) -> ConfigDict:
    """Linear grid search configuration over a list of parameters.

    Returns a configuration object that can be used to perform a grid search
    over a list of parameters. The parameters are sampled linearly between the
    minimum and maximum value.

    Args:
        param_names (list[str] | str): The name of the parameters to be
            sampled.
        min_values (list[float] | float): The minimum value of the parameters.
        max_values (list[float] | float): The maximum value of the parameters.
        n_steps (list[int] | int): The number of steps to take between the
            minimum and maximum value.

    Returns:
        ConfigDict: The configuration object that can be used to perform a grid
            search.
    """
    if isinstance(param_names, str):
        param_names = [param_names]
    if isinstance(min_values, float):
        min_values = [min_values]
    if isinstance(max_values, float):
        max_values = [max_values]
    if isinstance(n_steps, int):
        n_steps = [n_steps]

    assert (
        len(param_names) == len(min_values) == len(max_values) == len(n_steps)
    )

    config = ConfigDict()
    config.method = "grid"
    config.sampling_args = []
    for name, min_value, max_value, n_step in zip(
        param_names, min_values, max_values, n_steps
    ):
        config.sampling_args.append(
            [
                name,
                class_config(
                    linspace_sampler,
                    min_value=min_value,
                    max_value=max_value,
                    n_steps=n_step,
                ),
            ]
        )
    return config
