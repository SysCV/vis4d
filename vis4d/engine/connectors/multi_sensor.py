"""Static data connector for multi-sensor dataset."""
from __future__ import annotations

import torch

from typing import NamedTuple

from torch import Tensor

from vis4d.common import ArgsType, DictStrArrNested
from vis4d.data.const import CommonKeys as K
from vis4d.data.typing import DictData

from .static import StaticDataConnector


class MultiSensorDataConnector(StaticDataConnector):
    """Data connector for multi-sensor dataset."""

    def __init__(
        self,
        *args: ArgsType,
        default_sensor: str,
        sensors: list[str],
        **kwargs: ArgsType,
    ) -> None:
        """Initializes multi-sensor data connector with all required sensors.

        Args:
            *args: Arguments to pass to the parent class.
            default_sensor (str): The default sensor to use.
            sensors (list[str]): List of all sensors to use.
            **kwargs: Keyword arguments to pass to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.default_sensor = default_sensor
        self.sensors = sensors

    def get_test_input(self, data: DictData) -> DictData:
        """Returns the test input for the model."""
        test_input_dict: DictData = {
            v: [] for _, v in self.connections["test"].items()
        }
        for sensor in self.sensors:
            for k, v in self.connections["test"].items():
                test_input_dict[v].append(data[sensor][k])

        for key in test_input_dict:
            if key in [K.images, K.seg_masks]:
                test_input_dict[key] = torch.cat(test_input_dict[key])
            elif key in [K.intrinsics, K.extrinsics]:
                test_input_dict[key] = torch.stack(test_input_dict[key])
            else:
                test_input_dict[key] = sum(test_input_dict[key], [])

        return test_input_dict

    def get_callback_input(
        self,
        mode: str,
        prediction: NamedTuple | DictData,
        data: DictData,
        cb_type: str = "",
    ) -> dict[str, Tensor | DictStrArrNested]:
        """Returns the kwargs that are passed to the callback.

        Args:
            mode (str): Unique string defining which 'mode' to load for
                visualization. This could be 'semantics', 'bboxes' or similar.
            prediction (DictData): The datadict (e.g. output from model) which
                contains all the model outputs.
            data (DictData): The datadict (e.g. from the dataloader) which
                contains all data that was loaded.
            cb_type (str): Current type of the trainer loop. This can be
                'train', 'test' or 'val'.

        Raises:
            ValueError: If the key could not be found in the data dict.

        Returns:
            dict[str, Tensor | DictStrArrayNested]: kwargs that are passed
                onto the callback.
        """
        if f"{mode}_{cb_type}" in self.connections["callbacks"]:
            mode = f"{mode}_{cb_type}"
        else:
            return {}  # No inputs registered for this callback cb_type

        clbk_dict = self.connections["callbacks"][mode]

        try:
            return self._get_inputs_for_pred_and_data(
                clbk_dict, prediction, data[self.default_sensor]
            )

        except ValueError as e:
            raise ValueError(
                f"Error while loading callback input for mode {mode}.", e
            ) from e
