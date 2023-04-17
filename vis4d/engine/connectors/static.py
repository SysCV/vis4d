"""Static data connector for static remapping of the keys."""
from __future__ import annotations

from typing import NamedTuple

from torch import Tensor

from vis4d.common import DictStrArrNested
from vis4d.common.dict import get_dict_nested
from vis4d.common.named_tuple import get_from_namedtuple

from vis4d.data.typing import DictData
from vis4d.engine.util import is_namedtuple

from vis4d.engine.connectors.base import (
    DataConnector,
    DataConnectionInfo,
    SourceKeyDescription,
)


class StaticDataConnector(DataConnector):
    """DataConnector with static remapping of the keys."""

    def __init__(self, connections: DataConnectionInfo) -> None:
        """Creates new DataConnector with the specified DataConnectionInfo.

        This data connector is used in the training / testing loop in order to
        provide input to the models / losses / evaluators and visualizers.

        It extracts the required data from a 'DictData' objects and passes it
        to the next component with the provided new key.

        Args:
            connections (DataConnectionInfo): DataConnectionInfo defining the
            static key remappings.



        Simple Example Configuration:

        >>> train = dict(images = "images", gt = "gt_images)
        >>> test = dict(images = "images")
        >>> loss = dict(box_pred = dict(key = "masks", source = "prediction"),
                        box_gt = dict(key = "masks", source = "data")
                    )
        >>> connector = StaticDataConnector(dict(train =  train, test = test, loss = loss))

        More elaborate Configuration containing different visualizer and
        evaluators.

        >>> train = dict(images = "images", gt = "gt_images)
        >>> test = dict(images = "images")
        >>> loss = dict(box_pred = dict(key = "masks", source = "prediction"),
                        box_gt = dict(key = "masks", source = "data")
                    )
        >>> vis = dict(
                mask_visualizer = dict(
                    images = dict(key = "images", source = "data"),
                    masks = dict(key = "masks", source = "prediction"))
                )
        >>> evaluators = dict(
                semnatic_evaluator = dict(
                    prediction = dict(key = "masks", source = "prediction"))
                    groundtruth = dict(key = "masks", source = "data"))
                )
        >>> final_connection_dict = dict(train =  train, test = test,
                                        loss = loss,
                                        vis = vis, evaluators = evaluators
                                    )

        """
        self.connections = connections

    def _get_inputs_for_pred_and_data(
        self,
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

    def get_train_input(
        self, data: DictData
    ) -> dict[str, Tensor | DictStrArrNested]:
        """Returns the kwargs that are passed to the model for training.

        Args:
            data (DictData): The datadict (e.g. from the dataloader) which
                contains all data that was loaded.

        Returns:
            dict[str, Tensor | DictStrArrayNested]: kwargs that are passed
                onto the model.
        """
        return {v: data[k] for k, v in self.connections["train"].items()}

    def get_test_input(
        self, data: DictData
    ) -> dict[str, Tensor | DictStrArrNested]:
        """Returns the kwargs that are passed to the model for testing.

        Args:
            data (DictData): The datadict (e.g. from the dataloader) which
                contains all data that was loaded.

        Returns:
            dict[str, Tensor | DictStrArrayNested]: kwargs that are passed
                onto the model.
        """
        return {v: data[k] for k, v in self.connections["test"].items()}

    def get_loss_input(
        self, prediction: DictData | NamedTuple, data: DictData
    ) -> dict[str, Tensor | DictStrArrNested]:
        """Returns the kwargs that are passed to the loss during training.

        Args:
            prediction (DictData): The datadict (e.g. output from model) which
                contains all the model outputs.
            data (DictData): The datadict (e.g. from the dataloader) which
                contains all data that was loaded.

        Returns:
            dict[str, Tensor | DictStrArrayNested]: kwargs that are passed
                onto the loss.
        """
        return self._get_inputs_for_pred_and_data(
            self.connections["loss"], prediction, data
        )

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
        if self.connections["callbacks"] is None:
            return {}  # No data connections registered for callbacks

        if f"{mode}_{cb_type}" in self.connections["callbacks"]:
            mode = f"{mode}_{cb_type}"

        if mode in self.connections["callbacks"]:
            clbk_dict = self.connections["callbacks"][mode]
        else:
            return {}  # No inputs registered for this callback cb_type

        try:
            return self._get_inputs_for_pred_and_data(
                clbk_dict, prediction, data
            )
        except ValueError as e:
            raise ValueError(
                f"Error while loading callback input for mode {mode}.", e
            ) from e
