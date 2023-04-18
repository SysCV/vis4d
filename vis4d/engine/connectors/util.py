"""Utility functions for the connectors module."""
from __future__ import annotations

from copy import deepcopy
from typing import NamedTuple, TypedDict

from torch import Tensor

from vis4d.common.dict import get_dict_nested
from vis4d.common.named_tuple import get_from_namedtuple, is_namedtuple
from vis4d.common.typing import DictStrArrNested
from vis4d.data.typing import DictData


class SourceKeyDescription(TypedDict):
    """Defines a data entry by providing the key and source of the data.

    Attributes:
        key (str): Key that is used to index data from the specified source
        source (str): Which datasource to choose from.
            Options are ['data', 'prediction'] where data referes to the
            output of the dataloader and prediction refers to the model
            output
    """

    key: str
    source: str


def remap_pred_keys(
    info: dict[str, SourceKeyDescription], parent_key: str
) -> dict[str, SourceKeyDescription]:
    """Remaps the key of a connection mapping to a new parent key.

    Args:
        info (SourceKeyDescription): Description to remap.
        parent_key (str): New parent_key to use.

    Returns:
        SourceKeyDescription: Description with new key.

    """
    info = deepcopy(info)

    for value in info.values():
        if value["source"] == "prediction":
            value["key"] = parent_key + "." + value["key"]
    return info


def data_key(key: str) -> SourceKeyDescription:
    """Returns a SourceKeyDescription with data as source.

    Args:
        key (str): Key to use for the data entry.

    Returns:
        SourceKeyDescription: A SourceKeyDescription with data as source.
    """
    return SourceKeyDescription(key=key, source="data")


def pred_key(key: str) -> SourceKeyDescription:
    """Returns a SourceKeyDescription with prediction as source.

    Args:
        key (str): Key to use for the data entry.

    Returns:
        SourceKeyDescription: A SourceKeyDescription with prediction as source.
    """
    return SourceKeyDescription(key=key, source="prediction")


def get_inputs_for_pred_and_data(
    connection_dict: dict[str, SourceKeyDescription],
    prediction: DictData | NamedTuple,
    data: DictData,
) -> dict[str, Tensor | DictStrArrNested]:
    """Extracts input data from the provided SourceKeyDescription.

    Args:
        connection_dict (dict[str, SourceKeyDescription]): Input Key
            description which is used to gather and remap data from the
            two data dicts.
        prediction (DictData): Dict containing the model prediction output.
        data (DictData):  Dict containing the dataloader output.

    Raises:
        ValueError: If the datasource is invalid.

    Returns:
        dict[str, Tensor | DictStrArrayNested]: Dict containing new kwargs
            consisting of new key name and data extracted from
            the data dicts.
    """
    out = {}
    for new_key_name, old_key_name in connection_dict.items():
        # Assign field from data
        if old_key_name["source"] == "data":
            if old_key_name["key"] not in data:
                raise ValueError(
                    f"Key {old_key_name['key']} not found in data dict."
                    f"Available keys: {data.keys()}"
                )
            out[new_key_name] = data[old_key_name["key"]]

        # Assign field from model prediction
        elif old_key_name["source"] == "prediction":
            if is_namedtuple(prediction):
                out[new_key_name] = get_from_namedtuple(
                    prediction, old_key_name["key"]  # type: ignore
                )
            else:
                old_key = old_key_name["key"]
                out[new_key_name] = get_dict_nested(
                    prediction, old_key.split(".")  # type: ignore
                )
        else:
            raise ValueError(
                f"Unknown data source {old_key_name['source']}."
                f"Available: [prediction, data]"
            )
    return out
