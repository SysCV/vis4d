"""NuScenes data connector."""
from typing import NamedTuple

import torch
from torch import Tensor

from vis4d.common.typing import ArgsType, DictStrArrNested
from vis4d.data.typing import DictData
from vis4d.engine.connectors import StaticDataConnector


class NuscDataConnector(StaticDataConnector):
    """NuScenes data connector."""

    def __init__(
        self, *args: ArgsType, sensors: list[str], **kwargs: ArgsType
    ):
        super().__init__(*args, **kwargs)
        self.sensors = sensors

    def get_test_input(
        self, data: DictData
    ) -> dict[str, Tensor | DictStrArrNested]:
        """Returns the test input for the model."""
        test_input_dict = {v: [] for _, v in self.connections["test"].items()}
        for cam in self.sensors:
            for k, v in self.connections["test"].items():
                test_input_dict[v].append(data[cam][0][k])

        test_input_dict["images"] = torch.cat(test_input_dict["images"])
        test_input_dict["intrinsics"] = torch.stack(
            test_input_dict["intrinsics"]
        )
        test_input_dict["extrinsics"] = torch.stack(
            test_input_dict["extrinsics"]
        )

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
                clbk_dict, prediction, data["CAM_FRONT"][0]
            )

        except ValueError as e:
            raise ValueError(
                f"Error while loading callback input for mode {mode}.", e
            ) from e
