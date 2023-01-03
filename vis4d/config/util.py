"""Utils to parse and initialize a configuration file."""
from __future__ import annotations

import importlib
import re
from collections.abc import Sequence
from typing import Any

from ml_collections import ConfigDict

# Most of these functions need to deal with unknown parameters and are
# therefore not strictly typed


def class_config(class_path: str, **kwargs: Any) -> ConfigDict:  # type: ignore
    """Creates a configuration which can be instantiated as a class.

    This function creates a configuration dict which can be passed to
    'instantiate_classes' to create a instance of the given class or functor.

    Example:
    >>> class_cfg_obj = class_config("your.module.Module", arg1="arg1", arg2=2)
    >>>
    >>> print(class_cfg_obj)
    >>> # Prints :
    >>> class_path: your.module.Module
    >>> init_args:
    >>>   arg1: arg1
    >>>   arg2: 2
    >>>
    >>> # instantiate object
    >>> inst_obj = instantiate_classes(class_cfg_obj)
    >>> print(type(inst_obj)) # -> Will print Module


    Args:
        class_path (str): _description_
        **kwargs (any): Kwargs to pass to the class consstructor.

    Returns:
        ConfigDict: _description_
    """
    if class_path is None or len(kwargs) == 0:
        return ConfigDict({"class_path": class_path})
    return ConfigDict(
        {"class_path": class_path, "init_args": ConfigDict(kwargs)}
    )


def pprints_config(data: ConfigDict) -> str:
    """Converts a Config Dict into a string with a .yaml like structure.

    This function differs from __repr__ of ConfigDict in that it will not
    encode python classes using binary formats but just prints the __repr__
    of these classes.

    Args:
        data (ConfigDict): Configuration dict to convert to string

    Returns:
        str: A string representation of the ConfigDict
    """
    return _pprints_config(data)


def _pprints_config(  # type: ignore
    data: ConfigDict | Any,
    prefix: str = "",
    n_indents: int = 1,
) -> str:
    """Converts a Config Dict into a string with a .yaml like structure.

    This is the recursive implementation of 'pprints_config' and will be called
    recursively for every element in the dict.

    This function differs from __repr__ of ConfigDict in that it will not
    encode python classes using binary formats but just prints the __repr__
    of these classes.


    Args:
        data (ConfigDict | Any): Configuration dict or object to convert to
            string
        prefix (str): Prefix to print on each new line
        n_indents (int): Number of spaces to append for each nester property.

    Returns:
        str: A string representation of the ConfigDict
    """
    string_repr = ""
    if not isinstance(data, (dict, ConfigDict, list, tuple, dict)):
        return str(data)

    string_repr += "\n"

    if isinstance(data, (ConfigDict, dict)):
        for key in data:
            value = data[key]
            string_repr += (
                prefix
                + key
                + ": "
                + _pprints_config(value, prefix=prefix + " " * n_indents)
            ) + "\n"

    elif isinstance(data, (list, tuple)):
        for value in data:
            string_repr += prefix + "- "
            if isinstance(value, (ConfigDict, dict)):
                string_repr += "\n"

            string_repr += (
                _pprints_config(value, prefix=prefix + " " + " " * n_indents)
                + "\n"
            )
        string_repr += " \n"  # Add newline after list for better readability.

    # Clean up some formatting issues using regex. Could be done better
    string_repr = re.sub("\n\n+", "\n", string_repr)
    return re.sub("- +\n +", "- ", string_repr)


def pprint_config(data: ConfigDict) -> None:
    """Pretty prints a configuration dict to the console.

    Args:
        data (ConfigDict): The Configuration dict to print.
    """
    print(pprints_config(data))


def instantiate_classes(data: ConfigDict) -> ConfigDict | Any:  # type: ignore
    """Instantiates all classes in a given ConfigDict.

    This function iterates over the configuration data and instantiates
    all classes. Class defintions are provided by a config dict that has
    the following structure:

    {
        'data_path': 'path.to.my.class.Class',
        'init_args': ConfigDict(
            {
                'arg1': 'value1',
                'arg2': 'value2',
            }
        )
    }

    Args:
        data (ConfigDict): The general configuration object.

    Returns:
        ConfigDict | Any: The ConfigDict with all classes intialized. If the
        top level element is a class config, the returned element will be
        the instantiated class.
    """
    instantiated = _instantiate_classes(data)
    return instantiated


def _get_index(data: Any) -> Sequence[int] | Any:  # type: ignore
    """Internal function to generate a Sequence of indexes for a given object.

    Example:
    >>> [data[idx] for idx in _get_index(data)]

    Args:
        data (Any): The data entry to get an index for.

    Returns:
        Sequence[int] | Any: Iterable that can be used to index the data entry
            using e.g. [data[idx] for idx in _get_index(data)]
    """
    if isinstance(data, (list, tuple)):
        return range(len(data))
    return data


def _instantiate_classes(data: ConfigDict | Any) -> ConfigDict | Any:  # type: ignore # pylint: disable=line-too-long
    """Instantiates all classes in a given ConfigDict, tuple, list or Any.

    This is the recursive implementation of the 'instantiate_classes'.

    This function iterates over the configuration data and instantiates
    all classes. Class defintions are provided by a config dict that has
    the following structure:

    {
        'data_path': 'path.to.my.class.Class',
        'init_args': ConfigDict(
            {
                'arg1': 'value1',
                'arg2': 'value2',
            }
        )
    }

    Args:
        data (ConfigDict): The general configuration object.

    Returns:
        ConfigDict | Any: The ConfigDict with all classes intialized. If the
        top level element is a class config, the returned element will be
        the instantiated class.
    """
    if not isinstance(data, (ConfigDict, list, tuple)):
        return data

    for key in _get_index(data):
        value = data[key]

        if isinstance(value, ConfigDict):
            # Allow to convert ConfigDict to Object
            with data.ignore_type():
                data[key] = _instantiate_classes(value)

        elif isinstance(value, (list)):
            for idx, v in enumerate(value):
                data[key][idx] = _instantiate_classes(v)

        elif isinstance(value, (tuple)):
            data[key] = tuple(
                _instantiate_classes(value[idx]) for idx in range(len(value))
            )

    # Instantiate classs
    if "class_path" in data and not isinstance(data["class_path"], ConfigDict):
        module_name, class_name = data["class_path"].rsplit(".", 1)
        init_args = data.get("init_args", {})
        module = importlib.import_module(module_name)
        # Instantiate class
        clazz = getattr(module, class_name)(**init_args)
        return clazz

    return data
