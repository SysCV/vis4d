"""Utils to parse and initialize a configuration file."""
from __future__ import annotations

import importlib

from ml_collections import ConfigDict


# kwargs = init.get("init_args", {})
# if not isinstance(args, tuple):
#     args = (args,)
# class_module, class_name = init["class_path"].rsplit(".", 1)
# module = __import__(class_module, fromlist=[class_name])
# args_class = getattr(module, class_name)
# return args_class(*args, **kwargs)
def class_config(class_path: str, **kwargs) -> ConfigDict:
    if class_path is None or len(kwargs) == 0:
        return ConfigDict({"class_path": class_path})
    return ConfigDict(
        {"class_path": class_path, "init_args": ConfigDict(kwargs)}
    )


def instantiate_classes(data: ConfigDict) -> ConfigDict:
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
        ConfigDict: The ConfigDict with all classes intialized.
    """
    instantiated = _instantiate_classes(data)
    return instantiated  # TODO, type checks


def _get_index(data):
    if isinstance(data, (list, tuple)):
        return range(len(data))
    return data


def _instantiate_classes(data: ConfigDict) -> ConfigDict | any:
    """Instatntiates all classes in a given ConfigDict.

    Args:
        data (ConfigDict): The general configuration object.

    Returns:
        ConfigDict | any: _description_

        TODO FIX TYPING
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
            print(value)
            for idx in range(len(value)):
                data[key][idx] = _instantiate_classes(value[idx])

    if "class_path" in data and not isinstance(data["class_path"], ConfigDict):
        module_name, class_name = data["class_path"].rsplit(".", 1)
        init_args = data.get("init_args", {})
        module = importlib.import_module(module_name)
        # Instantiate class
        clazz = getattr(module, class_name)(**init_args)
        return clazz

    return data
