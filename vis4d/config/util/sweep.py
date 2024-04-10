"""Helper functions for creating sweep configurations."""

from __future__ import annotations

from ml_collections import ConfigDict

from vis4d.common.typing import ArgsType


def grid_search(
    param_names: list[str] | str,
    param_values: list[ArgsType] | list[list[ArgsType]],
) -> ConfigDict:
    """Linear grid search configuration over a list of parameters.

    Returns a configuration object that can be used to perform a grid search
    over a list of parameters.

    Args:
        param_names (list[str] | str): The name of the parameters to be
            sampled.
        param_values (list[Any] | list[list[Any]]): The values which
            should be sampled.

    Example:
        >>> grid_search("params.lr", [0.001, 0.01, 0.1])


    Returns:
        ConfigDict: The configuration object that can be used to perform a grid
            search.
    """
    if isinstance(param_names, str):
        param_names = [param_names]
        param_values = [param_values]

    assert len(param_names) == len(param_values)

    config = ConfigDict()
    config.method = "grid"
    config.sampling_args = []
    for name, values in zip(param_names, param_values):
        config.sampling_args.append([name, values])
    return config
