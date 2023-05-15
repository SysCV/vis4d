"""Data connector for multi-sensor dataset."""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from vis4d.common import ArgsType
from vis4d.data.typing import DictData

from .data_connector import DataConnector
from .util import SourceKeyDescription, get_field_from_prediction


class MultiSensorDataConnector(DataConnector):
    """Data connector for multi-sensor dataset."""

    def __init__(
        self,
        *args: ArgsType,
        sensors: list[str],
        **kwargs: ArgsType,
    ) -> None:
        """Initializes multi-sensor data connector with all required sensors.

        Args:
            *args: Arguments to pass to the parent class.
            sensors (list[str]): List of all sensors to use.
            **kwargs: Keyword arguments to pass to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.sensors = sensors

    def get_train_input(self, data: DictData) -> DictData:
        """Returns the train input for the model."""
        if self.train is None:
            return {}  # No data connections registered for training

        train_input_dict: DictData = {k: [] for k in self.train}
        for sensor in self.sensors:
            for k, v in self.train.items():
                train_input_dict[k].append(data[sensor][v])

        for key in train_input_dict:
            if isinstance(train_input_dict[key][0], Tensor):
                train_input_dict[key] = torch.stack(train_input_dict[key])
            else:
                train_input_dict[key] = sum(train_input_dict[key], [])

        return train_input_dict

    def get_test_input(self, data: DictData) -> DictData:
        """Returns the test input for the model."""
        if self.test is None:
            return {}  # No data connections registered for testing

        test_input_dict: DictData = {k: [] for k in self.test}
        for sensor in self.sensors:
            for k, v in self.test.items():
                test_input_dict[k].append(data[sensor][v])

        for key in test_input_dict:
            if isinstance(test_input_dict[key][0], Tensor):
                test_input_dict[key] = torch.stack(test_input_dict[key])
            else:
                test_input_dict[key] = sum(test_input_dict[key], [])

        return test_input_dict

    def get_loss_input(
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
        if self.loss is None:
            return {}  # No data connections registered for loss

        return get_multi_sensor_inputs(
            self.loss, prediction, data, self.sensors
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
                out[new_key_name] = sum(multi_sensor_data, [])

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
