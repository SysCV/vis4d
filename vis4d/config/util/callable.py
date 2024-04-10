"""Callable objects for use in config files."""

from ml_collections import ConfigDict

from vis4d.common.typing import ArgsType, GenericFunc
from vis4d.config import class_config, delay_instantiation


def get_callable_cfg(func: GenericFunc, **kwargs: ArgsType) -> ConfigDict:
    """Return callable config.

    Args:
        func (GenericFunc): Callable object.
        **kwargs (ArgsType): Keyword arguments to pass to the callable.

    Returns:
        ConfigDict: Config for the callable.
    """
    return delay_instantiation(class_config(func, **kwargs))
