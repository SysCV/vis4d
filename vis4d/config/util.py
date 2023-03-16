"""Utils to parse and initialize a configuration file."""
from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from ml_collections import ConfigDict as _ConfigDict
from ml_collections import FieldReference, FrozenConfigDict


# Most of these functions need to deal with unknown parameters and are
# therefore not strictly typed
class ConfigDict(_ConfigDict):  # type: ignore # pylint: disable=too-many-instance-attributes, line-too-long
    """A configuration dict which allows to access fields via dot notation.

    This class is a subclass of ml_collections._ConfigDict and overwrites the
    dot notation to return a FieldReference instead of a dict.

    For more information on the _ConfigDict class, see:
        ml_collections._ConfigDict

    Examples of using the ref and value mode:
        >>> config = _ConfigDict({"a": 1, "b": 2})
        >>> type(config.a)
        <class 'ml_collections.field_reference.FieldReference'>
        >>> config.valueMode() # Set the config to return values
        >>> type(config.a)
        <class 'int'>
    """

    def __init__(  # type: ignore
        self,
        initial_dictionary: Mapping[str, Any] | None = None,
        type_safe: bool = True,
        convert_dict: bool = True,
    ):
        """Creates an instance of _ConfigDict.

        Args:
          initial_dictionary: May be one of the following:

            1) dict. In this case, all values of initial_dictionary that are
            dictionaries are also be converted to _ConfigDict. However,
            dictionaries within values of non-dict type are untouched.

            2) _ConfigDict. In this case, all attributes are uncopied, and only
            the top-level object (self) is re-addressed. This is the same
            behavior as Python dict, list, and tuple.

            3) FrozenConfigDict. In this case, initial_dictionary is converted
            to a _ConfigDict version of the initial dictionary for the
            FrozenConfigDict (reversing any mutability changes FrozenConfigDict
            made).

          type_safe: If set to True, once an attribute value is assigned, its
            type cannot be overridden without .ignore_type() context manager.

          convert_dict: If set to True, all dict used as value in the
            _ConfigDict will automatically be converted to _ConfigDict.
        """
        super().__init__(initial_dictionary, type_safe, convert_dict)
        object.__setattr__(self, "_return_refs", True)

    def set_ref_mode(self, ref_mode: bool) -> None:
        """Sets the config to return references instead of values."""

        def _rec_resolve_iterable(  # type: ignore
            iterable: Iterable[Any], cfgs: list[ConfigDict]
        ) -> None:
            """Recursively adds all ConfigDicts to a list."""
            for item in iterable:
                if isinstance(item, ConfigDict):
                    cfgs.append(item)
                elif isinstance(item, (list, tuple)):
                    _rec_resolve_iterable(item, cfgs)
                elif isinstance(item, (dict, _ConfigDict)):
                    _rec_resolve_iterable(item.values(), cfgs)

        # Update value of this dict
        object.__setattr__(self, "_return_refs", ref_mode)

        # propagate to sub configs
        for value in self.values():
            if isinstance(value, ConfigDict):
                value = value.value_mode()
            elif isinstance(value, (list, tuple, ConfigDict, dict)):
                cfgs: list[ConfigDict] = []
                _rec_resolve_iterable(value, cfgs)
                for cfg in cfgs:
                    cfg.set_ref_mode(ref_mode)

    def ref_mode(self) -> ConfigDict:
        """Sets the config to return references instead of values."""
        self.set_ref_mode(True)
        return self

    def value_mode(self) -> ConfigDict:
        """Sets the config to return values instead of references."""
        self.set_ref_mode(False)
        return self

    def __getitem__(self, key: str) -> FieldReference:
        """Returns the reference for the given key."""
        # private properties are always returned as values

        if self._return_refs:
            try:
                return super().get_ref(key)
            except ValueError:
                pass
        return super().__getitem__(key)


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


def class_config(clazz: type | Callable[Any, Any] | str, **kwargs: Any) -> _ConfigDict:  # type: ignore # pylint: disable=line-too-long
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
        _ConfigDict: _description_
    """
    class_path = resolve_class_name(clazz)
    if class_path is None or len(kwargs) == 0:
        return _ConfigDict({"class_path": class_path})
    return _ConfigDict(
        {"class_path": class_path, "init_args": _ConfigDict(kwargs)}
    )


def pprints_config(data: _ConfigDict) -> str:
    """Converts a Config Dict into a string with a .yaml like structure.

    This function differs from __repr__ of _ConfigDict in that it will not
    encode python classes using binary formats but just prints the __repr__
    of these classes.

    Args:
        data (_ConfigDict): Configuration dict to convert to string

    Returns:
        str: A string representation of the _ConfigDict
    """
    return _pprints_config(data)


def _pprints_config(  # type: ignore
    data: _ConfigDict | Any,
    prefix: str = "",
    n_indents: int = 1,
) -> str:
    """Converts a Config Dict into a string with a .yaml like structure.

    This is the recursive implementation of 'pprints_config' and will be called
    recursively for every element in the dict.

    This function differs from __repr__ of _ConfigDict in that it will not
    encode python classes using binary formats but just prints the __repr__
    of these classes.


    Args:
        data (_ConfigDict | Any): Configuration dict or object to convert to
            string
        prefix (str): Prefix to print on each new line
        n_indents (int): Number of spaces to append for each nester property.

    Returns:
        str: A string representation of the _ConfigDict
    """
    string_repr = ""
    if isinstance(data, FieldReference):
        data = data.get()

    if not isinstance(data, (dict, _ConfigDict, list, tuple, dict)):
        return str(data)

    string_repr += "\n"

    if isinstance(data, (_ConfigDict, dict)):
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
            if isinstance(value, (_ConfigDict, dict)):
                string_repr += "\n"

            string_repr += (
                _pprints_config(value, prefix=prefix + " " + " " * n_indents)
                + "\n"
            )
        string_repr += " \n"  # Add newline after list for better readability.

    # Clean up some formatting issues using regex. Could be done better
    string_repr = re.sub("\n\n+", "\n", string_repr)
    return re.sub("- +\n +", "- ", string_repr)


def pprint_config(data: _ConfigDict) -> None:
    """Pretty prints a configuration dict to the console.

    Args:
        data (_ConfigDict): The Configuration dict to print.
    """
    print(pprints_config(data))


def delay_instantiation(instantiable: _ConfigDict) -> _ConfigDict:
    """Delays the instantiation of the given configuration object.

    This is a somewhat hacky way to delay the initialization of the optimizer
    configuration object. It works by replacing the class_path with _class_path
    which basically tells the instantiate_classes function to not instantiate
    the class. Instead, it returns a function that can be called to instantiate
    the class

    Args:
        instantiable (_ConfigDict): The configuration object to delay the
            instantiation of.
    """
    instantiable["_class_path"] = instantiable["class_path"]
    del instantiable["class_path"]

    return class_config(DelayedInstantiator, instantiable=instantiable)


class DelayedInstantiator:
    """Class that delays the instantiation of the given configuration object.

    This is a somewhat hacky way to delay the initialization of the optimizer
    configuration object. It works by replacing the class_path with _class_path
    which basically tells the instantiate_classes function to not instantiate
    the class. Instead, it returns a function that can be called to instantiate
    the class.

    Args:
        instantiable (_ConfigDict): The configuration object to delay the
            instantiation of.
    """

    def __init__(self, instantiable: _ConfigDict) -> None:
        """Instantiates the DelayedInstantiator."""
        self.instantiable = copy_and_resolve_references(instantiable)

    def __call__(self, **kwargs: Any) -> Any:  # type: ignore
        """Instantiates the configuration object."""
        for k, v in kwargs.items():
            self.instantiable["init_args"][k] = v
        self.instantiable["class_path"] = self.instantiable["_class_path"]

        del self.instantiable["_class_path"]
        ins = instantiate_classes(self.instantiable)
        return ins


def instantiate_classes(data: _ConfigDict) -> _ConfigDict | Any:  # type: ignore # pylint: disable=line-too-long
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
        data (_ConfigDict): The general configuration object.

    Returns:
        _ConfigDict | Any: The _ConfigDict with all classes intialized. If the
        top level element is a class config, the returned element will be
        the instantiated class.
    """
    resolved = copy_and_resolve_references(data)
    instantiated_objects = _instantiate_classes(resolved)
    return instantiated_objects


def copy_and_resolve_references(  # type: ignore
    config: _ConfigDict | Any, visit_map: dict[int, Any] | None = None
):
    """Returns a _ConfigDict copy with FieldReferences replaced by values.

    If the object is a FrozenConfigDict, the copy returned is also a
    FrozenConfigDict. However, note that FrozenConfigDict should already have
    FieldReferences resolved to values, so this method effectively produces
    a deep copy.

    Note: This method is overwritten from the _ConfigDict class and allows to
    also resolve FieldReferences in lists and tuples.

    Args:
        config: _ConfigDict object to copy.
        visit_map: A mapping from _ConfigDict object ids to their copy. Method
            is recursive in nature, and it will call
            ".copy_and_resolve_references(visit_map)" on each encountered
            object, unless it is already in visit_map.


    Returns:
        _ConfigDict copy with previous FieldReferences replaced by values.
    """
    if isinstance(config, FieldReference):
        config = config.get()

    if isinstance(config, (list, tuple)):
        return type(config)(
            copy_and_resolve_references(value, visit_map) for value in config
        )
    if not isinstance(config, _ConfigDict):
        return config

    visit_map = visit_map or {}
    config_dict_copy = ConfigDict()
    config_dict_copy.value_mode()
    super(_ConfigDict, config_dict_copy).__setattr__(
        "_convert_dict", config.convert_dict
    )
    visit_map[id(config)] = config_dict_copy

    for key, value in config._fields.items():
        if isinstance(value, FieldReference):
            value = value.get()

        if id(value) in visit_map:
            value = visit_map[id(value)]
        elif isinstance(value, _ConfigDict):
            value = copy_and_resolve_references(value, visit_map)
        elif isinstance(value, list):
            value = [copy_and_resolve_references(v, visit_map) for v in value]
        elif isinstance(value, tuple):
            value = tuple(
                copy_and_resolve_references(v, visit_map) for v in value
            )

        if isinstance(config, FrozenConfigDict):
            config_dict_copy._frozen_setattr(  # pylint:disable=protected-access
                key, value
            )
        else:
            config_dict_copy[key] = value

    super(_ConfigDict, config_dict_copy).__setattr__(
        "_locked", config.is_locked
    )
    super(_ConfigDict, config_dict_copy).__setattr__(
        "_type_safe", config.is_type_safe
    )
    return config_dict_copy


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


def _instantiate_classes(data: _ConfigDict | Any) -> ConfigDict | Any:  # type: ignore # pylint: disable=line-too-long
    """Instantiates all classes in a given _ConfigDict, tuple, list or Any.

    This is the recursive implementation of the 'instantiate_classes'.

    This function iterates over the configuration data and instantiates
    all classes. Class defintions are provided by a config dict that has
    the following structure:

    {
        'data_path': 'path.to.my.class.Class',
        'init_args': _ConfigDict(
            {
                'arg1': 'value1',
                'arg2': 'value2',
            }
        )
    }

    Args:
        data (_ConfigDict): The general configuration object.

    Returns:
        _ConfigDict | Any: The _ConfigDict with all classes intialized. If the
        top level element is a class config, the returned element will be
        the instantiated class.
    """
    if isinstance(data, FieldReference):
        data = data.get()

    if not isinstance(data, (_ConfigDict, list, tuple)):
        return data

    for key in _get_index(data):
        value = data[key]
        if isinstance(value, FieldReference):
            value = value.get()
        # resolve field refs
        if isinstance(value, _ConfigDict):
            # Allow to convert _ConfigDict to Object
            if isinstance(data, _ConfigDict):
                with data.ignore_type():
                    data[key] = _instantiate_classes(value)
            else:
                data[key] = _instantiate_classes(value)

        elif isinstance(value, (list)):
            for idx, v in enumerate(value):
                data[key][idx] = _instantiate_classes(v)

        elif isinstance(value, (tuple)):
            data[key] = tuple(
                _instantiate_classes(value[idx]) for idx in range(len(value))
            )

    # Instantiate classs
    if "class_path" in data and not isinstance(
        data["class_path"], _ConfigDict
    ):
        module_name, class_name = data["class_path"].rsplit(".", 1)
        init_args = data.get("init_args", {})
        module = importlib.import_module(module_name)
        # Instantiate class
        clazz = getattr(module, class_name)(**init_args)
        return clazz

    return data
