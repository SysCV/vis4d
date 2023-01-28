"""This module contains dictionary utility functions."""
from __future__ import annotations

from typing import Any

from vis4d.common import DictStrAny


def get_dict_nested(  # type: ignore
    dictionary: DictStrAny, keys: list[str]
) -> Any:
    """Get a value from a nested dictionary.

    Args:
        dictionary (DictStrAny): The dictionary to get the value from.
        keys (list[str]): A list of keys specifying the location in the nested
            dictionary where the value is located.

    Returns:
        list[str]: The value from the dictionary.

    Raises:
        ValueError: If the key is not present in the dictionary.

    Examples:
        >>> d = {'a': {'b': {'c': 10}}}
        >>> get_dict_nested(d, ['a', 'b', 'c'])
        10

        >>> get_dict_nested(d, ['a', 'b', 'd'])
        ValueError: Key d not in dictionary! Current keys: dict_keys(['c'])
    """
    for key in keys:
        if key not in dictionary:
            raise ValueError(
                f"Key {key} not in dictionary! Current keys: "
                f"{dictionary.keys()}"
            )
        dictionary = dictionary[key]
    return dictionary


def set_dict_nested(  # type: ignore
    dictionary: DictStrAny, keys: list[str], value: Any
) -> None:
    """Set a value in a nested dictionary.

    Args:
        dictionary (dict[str, Any]): The dictionary to set the value in.
        keys (list[str]): A list of keys specifying the location in the nested
            dictionary where the value should be set.
        value (Any): The value to set in the dictionary.

    Examples:
        >>> d = {}
        >>> set_dict_nested(d, ['a', 'b', 'c'], 10)
        >>> d
        {'a': {'b': {'c': 10}}}
    """
    for key in keys[:-1]:
        dictionary = dictionary.setdefault(key, {})
    dictionary[keys[-1]] = value
