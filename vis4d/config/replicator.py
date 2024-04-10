"""Replication methods to perform different parameters sweeps."""

from __future__ import annotations

import re
from collections.abc import Callable, Generator, Iterable
from queue import Queue
from typing import Any

from ml_collections import ConfigDict

from vis4d.common.typing import ArgsType


def iterable_sampler(  # type: ignore
    samples: Iterable[Any],
) -> Callable[[], Generator[Any, None, None]]:
    """Creates a sampler from an iterable.

    This fuction returns a method that returns a generator that iterates
    over all values provided in the 'samples' iterable.

    Args:
        samples (Iterable[Any]): Iterable over which to sample.

    Returns:
        Callable[[], Generator[Any, None, None]]: Function that
            returns a generator which iterates over all elements in the given
            iterable.
    """

    def _sampler() -> Generator[float, None, None]:
        yield from samples

    return _sampler


def linspace_sampler(
    min_value: float, max_value: float, n_steps: int = 1
) -> Callable[[], Generator[float, None, None]]:
    """Creates a linear space sampler.

    This fuction returns a method that returns a generator that iterates
    from min_value to max_value in n_steps.

    Args:
        min_value (float): Lower value bound
        max_value (float): Upper value bound
        n_steps (int, optional): Number of steps. Defaults to 1.

    Returns:
        Callable[[], Generator[float, None, None]]: Function that
            returns a generator which iterates from min to max in n_steps.
    """

    def _sampler() -> Generator[float, None, None]:
        for i in range(n_steps):
            yield min_value + i / max(n_steps - 1, 1) * (max_value - min_value)

    return _sampler


def logspace_sampler(
    min_exponent: float,
    max_exponent: float,
    n_steps: int = 1,
    base: float = 10,
) -> Callable[[], Generator[float, None, None]]:
    """Creates a logarithmic space sampler.

    This fuction returns a method that returns a generator that iterates
    from base^min_exponent to base^max_exponent in n_steps.

    Args:
        min_exponent (float): Lower value bound
        max_exponent (float): Upper value bound
        n_steps (int, optional): Number of steps. Defaults to 1.
        base (float): Base value for exponential calculation. Defaults to 10.

    Returns:
        Callable[[], Generator[float, None, None]]: Function that
           returns a generator which iterates from 10^min to 10^max in n_steps.
    """

    def _sampler() -> Generator[float, None, None]:
        for exp in linspace_sampler(min_exponent, max_exponent, n_steps)():
            yield base**exp

    return _sampler


def replicate_config(  # type: ignore
    configuration: ConfigDict,
    sampling_args: list[
        tuple[str, Callable[[], Generator[Any, None, None]] | Iterable[Any]]
    ],
    method: str = "grid",
    fstring="",
) -> Generator[ConfigDict, None, None]:
    """Function used to replicate a config.

    This function takes a ConfigDict and a dict with (key: generator) entries.
    It will yield, multiple modified config dicts assigned with different
    values defined in the sampling_args dictionary.

    Example:
    >>> config = ConfigDict({"trainer": {"lr": 0.2, "bs": 2}})
    >>> replicated_config = replicate_config(config,
    >>>         sampling_args = [("trainer.lr", linspace_sampler(0.01, 0.1, 3))],
    >>>         method = "grid"
    >>>     )
    >>> for c in replicated_config:
    >>>     print(c)

    Will print:
        trainer: bs: 2 lr: 0.01
        trainer: bs: 2 lr: 0.055
        trainer: bs: 2 lr: 0.1

    NOTE, the config dict instance that will be returned will be mutable and
    continuously updated to preserve references.
    In the code above, executing
    >>> print(list(replicated_config))
    Prints:
        trainer: bs: 2 lr: 0.1
        trainer: bs: 2 lr: 0.1
        trainer: bs: 2 lr: 0.1

    Please resolve the reference and copy the dict if you need a list:
    >>> print([c.copy_and_resolve_references() for c in replicated_config])


    Args:
        configuration (ConfigDict): Configuration to replicate
        sampling_args (dict[str, Callable[[], Any]]): The queue,
            that contains (key, iterator) pairs where the iterator
            yields the values which should be assigned to the key.
        method (str): What replication method to use. Currently only
            'grid' and 'linear' is supported.
            Grid combines the sampling arguments in a grid wise fashion
            ([1,2],[3,4] -> [1,3],[1,4],[2,3],[2,4]) whereas 'linear' will
            only select elements at the same index ([1,2],[3,4]->[1,3],[2,4]).
        fstring (str): Format string to use for the experiment name. Defaults
            to an empty string. The format string will be resolved with the
            values of the config dict. For example, if the config dict
            contains a key 'trainer.lr' with value 0.1, the format string
            '{trainer.lr}' will be resolved to '0.1'.

    Raises:
        ValueError: if the replication method is unknown.
    """
    sampling_queue: Queue[  # type: ignore
        tuple[str, Callable[[], Generator[Any, None, None]]]
    ] = Queue()

    for key, value in sampling_args:
        # Convert Iterable to a callable generator
        if isinstance(value, Iterable):
            generator: Callable[[], Generator[ArgsType, None, None]] = (
                lambda value=value: (i for i in value)  # type: ignore
            )
            sampling_queue.put((key, generator))
        else:
            sampling_queue.put((key, value))

    if method == "grid":
        replicated = _replicate_config_grid(configuration, sampling_queue)
    elif method == "linear":
        replicated = _replicate_config_linear(configuration, sampling_queue)
    else:
        raise ValueError(f"Unknown replication method {method}")

    original_name = configuration.experiment_name

    for config in replicated:
        # Update config name
        config.experiment_name = (
            f"{original_name}_{_resolve_fstring(fstring, config)}"
        )
        yield config


def _resolve_fstring(fstring: str, config: ConfigDict) -> str:
    """Resolves a format string with the values from the config.

    This function takes a format string and replaces all the keys
    with the values from the config. The keys are expected to be
    in the format {key} or {key:format}.
    This function may fail if the format string contains a key that
    is not present in the config. It will also fail if the format
    string contains a key that is not a valid python identifier.

    Args:
        fstring (str): The format string. E.g. "lr_{params.lr}".
        config (ConfigDict): The config dict. E.g. {"params": {"lr": 0.1}}.

    Returns:
        str: The resolved format string. E.g. "lr_0.1
    """
    # match everything between { and ':' or '}'
    pattern = re.compile(r"{([^:}]+)")
    required_params = {p.strip() for p in pattern.findall(fstring)}

    format_dict: dict[str, str] = {}
    for param in required_params:
        # Maks out '.' which is invalid for .format() call
        new_param_name = param.replace(".", "_")
        format_dict[new_param_name] = getattr(config, param)
        fstring = fstring.replace(param, new_param_name)

    # apply formatting
    return fstring.format(**format_dict)


def _replicate_config_grid(  # type: ignore
    configuration: ConfigDict,
    sampling_queue: Queue[
        tuple[str, Callable[[], Generator[Any, None, None]]]
    ],
) -> Generator[ConfigDict, None, None]:
    """Internal function used to replicate a config.

    This function takes a ConfigDict and a queue with (key, generator) entries.
    It will then recursively call itself and yield the ConfigDict with
    updated values for every key in the sampling_queue. Each key combination
    will be yielded exactly once, resulting in prod(len(generator)) entires.


    For example, a parameter sweep using 'lr: [0,1], bs: [8, 16]' will yield
    [0, 8], [0, 16], [0, 8], [1, 16] as combinations.

    Args:
        configuration (ConfigDict): Configuration to replicate
        sampling_queue (Queue[tuple[str, Callable[[], Any]]]): The queue,
            that contains (key, iterator) pairs where the iterator
            yields the values which should be assigned to the key.

    Yields:
        ConfigDict: Replicated configuration with updated key values.
    """
    # Termination criterion reached, We processed all samplers
    if sampling_queue.empty():
        yield configuration
        return

    # Get next key we want to replicate
    (key_name, sampler) = sampling_queue.get()

    # Iterate over all possible assignement values for this key
    for value in sampler():
        # Update value ignoring type errors
        # (e.g. user set default lr to 1 instead 1.0 would
        # otherwise give a type error (float != int))
        with configuration.ignore_type():
            configuration.update_from_flattened_dict({key_name: value})

        # Let the other samplers process the remaining keys
        yield from _replicate_config_grid(configuration, sampling_queue)

    # Add back this sampler for next round
    sampling_queue.put((key_name, sampler))


def _replicate_config_linear(  # type: ignore
    configuration: ConfigDict,
    sampling_queue: Queue[
        tuple[str, Callable[[], Generator[Any, None, None]]]
    ],
    current_position: int | None = None,
) -> Generator[ConfigDict, None, None]:
    """Internal function used to replicate a config in a linear way.

    This function takes a ConfigDict and a queue with (key, generator) entries.
    It will then recursively call itself and yield the ConfigDict with
    updated values for every key in the sampling_queue.

    For example, a parameter sweep using 'lr: [0,1], bs: [8, 16]' will yield
    [0, 8], [1, 16] as combinations.

    Args:
        configuration (ConfigDict): Configuration to replicate
        sampling_queue (Queue[tuple[str, Callable[[], Any]]]): The queue,
            that contains (key, iterator) pairs where the iterator
            yields the values which should be assigned to the key.
        current_position (int, optional): Current position of the top level
            sampling module. Used and updated internally.

    Yields:
        ConfigDict: Replicated configuration with updated key values.
    """
    # Termination criterion reached, We processed all samplers
    if sampling_queue.empty():
        yield configuration
        return

    # Get next key we want to replicate
    (key_name, sampler) = sampling_queue.get()

    is_top_level = False
    if current_position is None:
        is_top_level = True  # This is the top level call.
        current_position = 0

    # Iterate over all possible assignement values for this key
    for idx, value in enumerate(sampler()):
        if not is_top_level and idx != current_position:
            continue  # only yield entry that matches requested position

        # Update value ignoring type errors
        # (e.g. user set default lr to 1 instead 1.0 would
        # otherwise give a type error (float != int))
        with configuration.ignore_type():
            configuration.update_from_flattened_dict({key_name: value})

        # Let the other samplers process the remaining keys
        yield from _replicate_config_linear(
            configuration, sampling_queue, current_position
        )

        if is_top_level:
            current_position += 1

    # Add back this sampler for next round
    sampling_queue.put((key_name, sampler))
