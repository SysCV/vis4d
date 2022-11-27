"""Dictionary utils."""
from __future__ import annotations

from typing import Any

from vis4d.common import DictStrAny


def get_dict_nested(  # type: ignore
    dictionary: DictStrAny, keys: list[str]
) -> Any:
    """Get value in nested dict."""
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
    """Set value in nested dict."""
    for key in keys[:-1]:
        dictionary = dictionary.setdefault(key, {})
    dictionary[keys[-1]] = value
