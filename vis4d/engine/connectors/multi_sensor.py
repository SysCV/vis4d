"""Data connector for multi-sensor dataset."""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from vis4d.common.typing import ArgsType, DictStrArrNested
from vis4d.data.typing import DictData

from .base import CallbackConnector, DataConnector, LossConnector
from .util import SourceKeyDescription, get_field_from_prediction


class MultiSensorDataConnector(DataConnector):
    """Data connector for multi-sensor data dict."""

    def __init__(
        self, *args: ArgsType, sensors: list[str], **kwargs: ArgsType
    ) -> None:
        """Initializes multi-sensor data connector with required sensors.

        Args:
            *args: Arguments to pass to the parent class.
            sensors (list[str]): List of all sensors to use.
            **kwargs: Keyword arguments to pass to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.sensors = sensors

    def __call__(self, data: DictData) -> DictData:
        """Returns the train input for the model."""
        input_dict: DictData = {}
        for k, v in self.key_mapping.items():
            input_dict[k] = [data[sensor][v] for sensor in self.sensors]

        for k, v in input_dict.items():
            if isinstance(v[0], Tensor):
                input_dict[k] = torch.stack(input_dict[k])
            else:
                input_dict[k] = input_dict[k]

        return input_dict


class MultiSensorLossConnector(LossConnector):
    """Multi-sensor Data connector for loss module of the training pipeline."""

    def __init__(
        self, *args: ArgsType, sensors: list[str], **kwargs: ArgsType
    ) -> None:
        """Initializes multi-sensor data connector with required sensors.

        Args:
            *args: Arguments to pass to the parent class.
            sensors (list[str]): List of all sensors to use.
            **kwargs: Keyword arguments to pass to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.sensors = sensors

    def __call__(
        self, prediction: DictData | NamedTuple, data: DictData
    ) -> DictData:
        """Returns the kwargs that are passed to the loss during training.

        Args:
            prediction (DictData): The datadict (e.g. output from model) which
                contains all the model outputs.
            data (DictData): The datadict (e.g. from the dataloader) which
                contains all data that was loaded.

        Returns:
            DictData: kwargs that are passed onto the loss.
        """
        return get_multi_sensor_inputs(
            self.key_mapping, prediction, data, self.sensors
        )


class MultiSensorCallbackConnector(CallbackConnector):
    """Multi-sensor data connector for the callback."""

    def __init__(
        self, *args: ArgsType, sensors: list[str], **kwargs: ArgsType
    ) -> None:
        """Initializes multi-sensor data connector with required sensors.

        Args:
            *args: Arguments to pass to the parent class.
            sensors (list[str]): List of all sensors to use.
            **kwargs: Keyword arguments to pass to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.sensors = sensors

    def __call__(
        self, prediction: DictData, data: DictData
    ) -> dict[str, Tensor | DictStrArrNested]:
        """Returns the kwargs that are passed to the callback during training.

        Args:
            prediction (DictData): The datadict (e.g. output from model) which
                contains all the model outputs.
            data (DictData): The datadict (e.g. from the dataloader) which
                contains all data that was loaded.

        Returns:
            dict[str, Tensor | DictStrArrNested]: kwargs that are passed
                onto the callback.
        """
        return get_multi_sensor_inputs(
            self.key_mapping, prediction, data, self.sensors
        )


def get_multi_sensor_inputs(
    connection_dict: dict[str, SourceKeyDescription],
    prediction: DictData | NamedTuple,
    data: DictData,
    sensors: list[str],
) -> DictData:
    """Extracts multi-sensor input data from the provided SourceKeyDescription.

    Args:
        connection_dict (dict[str, SourceKeyDescription]): Input Key
            description which is used to gather and remap data from the
            two data dicts.
        prediction (DictData): Dict containing the model prediction output.
        data (DictData):  Dict containing the dataloader output.
        sensors (list[str]): List of all sensors to use.

    Raises:
        ValueError: If the datasource is invalid.

    Returns:
        out (DictData): Dict containing new kwargs consisting of new key name
            and data extracted from the data dicts.
    """
    out: DictData = {}
    for new_key_name, old_key_name in connection_dict.items():
        # Assign field from data
        if old_key_name["source"] == "data":
            multi_sensor_data = [
                data[sensor][old_key_name["key"]] for sensor in sensors
            ]

            if isinstance(multi_sensor_data[0], Tensor):
                out[new_key_name] = torch.stack(multi_sensor_data)
            else:
                out[new_key_name] = multi_sensor_data

        # Assign field from prediction
        elif old_key_name["source"] == "prediction":
            out[new_key_name] = get_field_from_prediction(
                prediction, old_key_name
            )
        else:
            raise ValueError(
                f"Unknown data source {old_key_name['source']}."
                f"Available: [prediction, data]"
            )
    return out
