"""Utils to parse and initialize a configuration file."""
from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Sequence
from typing import Any

from ml_collections import ConfigDict

# Most of these functions need to deal with unknown parameters and are
# therefore not strictly typed


def resolve_class_name(clazz: type | Callable[Any, Any] | str) -> str:  # type: ignore # pylint: disable=line-too-long
    """Resolves the full class name of the given class object, callable or str.

    This function takes a class object and returns the class name as a string.
    Args:
        clazz (type | Callable[[Any], Any] | str): The object to resolve
            the full path of.

    Returns:
        str: The full path of the given object.

    Raises:
        ValueError: If the given object is a lambda function.

    Examples:
        >>> class MyClass: pass
        >>> resolve_class_name(MyClass)
        '__main__.MyClass'
        >>> resolve_class_name("path.to.MyClass")
        'path.to.MyClass'
        >>> def my_function(): pass
        >>> resolve_class_name(my_function)
        '__main__.my_function'
    """
    if isinstance(clazz, str):
        return clazz

    if clazz.__name__ == "lambda":
        raise ValueError(
            "Resolving the full class path of lambda functions"
            "is not supported. Please define a inline function instead."
        )

    module = clazz.__module__
    if module is None or module == str.__class__.__module__:
        return clazz.__name__
    return module + "." + clazz.__name__


def class_config(clazz: type | Callable[Any, Any] | str, **kwargs: Any) -> ConfigDict:  # type: ignore # pylint: disable=line-too-long
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
    >>> print(type(inst_obj)) # -> Will print <class 'your.module.Module'>

    >>> # Example by directly passing objects:
    >>> class MyClass:
    >>>     def __init__(self, name: str, age: int):
    >>>         self.name = name
    >>>         self.age = age
    >>> class_cfg_obj = class_config(MyClass, name="John", age= 25)

    >>> print(class_cfg_obj)
    >>> # Prints :
    >>> class_path: __main__.MyClass
    >>> init_args:
    >>>   name: John
    >>>   age: 25

    >>> # instantiate object
    >>> inst_obj = instantiate_classes(class_cfg_obj)
    >>> print(type(inst_obj)) # -> Will print <class '__main__.MyClass'>
    >>> print(inst_obj.name) # -> Will print John

    Args:
        clazz (type | Callable[[Any], Any] | str): class type or functor or
            class string path.
        **kwargs (any): Kwargs to pass to the class constructor.

    Returns:
        ConfigDict: _description_
    """
    class_path = resolve_class_name(clazz)
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


def delayed_instantiator(instantiable: ConfigDict) -> Any:  # type: ignore
    """Class that delays the instantiation of the given configuration object.

    This is a somewhat hacky way to delay the initialization of the optimizer
    configuration object. It works by replacing the class_path with _class_path
    which basically tells the instantiate_classes function to not instantiate
    the class. Instead, it returns a function that can be called to instantiate
    the class.

    TODO: Look for a better way to do this.

    Args:
        instantiable (ConfigDict): The configuration object to delay the
            instantiation of.

    Example:
    >>> config = ConfigDict()
    >>> config.class_path = "torch.optim.Adam"
    >>> config.init_args = ConfigDict()
    >>> config.init_args.lr = 0.001
    >>> config.init_args.betas = (0.9, 0.999)

    >>> optimizer_cb = instantiate_classes(delayed_instantiator(config))
    type(optimizer_cb) # -> <class 'function'>, not torch.optim.Adam

    >> # Later when the optimizer is needed, it can be instantiated with:
    >> optimizer = optimizer_cb(params = [])
    type(optimizer) # -> torch.optim.Adam
    """
    # Make instantiable non instantiable
    instantiable["_class_path"] = instantiable["class_path"]
    del instantiable["class_path"]

    def delayed_callable(**kwargs) -> Any:  # type: ignore
        """Instantiates the configuration object."""
        for k, v in kwargs.items():
            instantiable["init_args"][k] = v
        instantiable["class_path"] = instantiable["_class_path"]
        del instantiable["_class_path"]

        return instantiate_classes(instantiable)

    return delayed_callable


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
