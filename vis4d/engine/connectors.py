"""Defines data structures for data connection."""
from __future__ import annotations

from typing import Dict, NamedTuple, TypedDict, Union

from torch import Tensor
from typing_extensions import NotRequired

from vis4d.common.dict import get_dict_nested
from vis4d.common.named_tuple import get_from_namedtuple
from vis4d.data.typing import DictData
from vis4d.engine.util import is_namedtuple

DictStrArrNested = Dict[str, Union[Tensor, Dict[str, Tensor]]]

### Type Definitions


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


class DataConnectionInfo(TypedDict):
    """Defines how and which entries to connect.

    This datastructure defines which kwargs to pass to the connected component.
    It consists of four sections:
    1. 'train / test': Defines which kwargs to pass onto the model.
    2. 'loss': Defines which kwargs to pass onto the loss. This field requres
        a key and a source for each entry since the loss takes data from the
        model output and data input.
    3. 'vis/evaluators', Optional: Defines which kwargs to use for
        the visualizer/evaluators. This data structure is slightly more
        complicated, since we could have mutliple evaluators / visualizers
        that might depend on model outputs and data from the dataloader.


    Attributes:
        train (dict[str, str]):
        test (dict[str, str]):

        loss (dict[str, SourceKeyDescription]):

        vis  (dict[str, dict[str, SourceKeyDescription]]):
        evaluators (dict[str, dict[str, SourceKeyDescription]]):



    Simple Example Configuration:

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

    train: dict[str, str]
    test: dict[str, str]

    loss: dict[str, SourceKeyDescription]

    callbacks: NotRequired[dict[str, dict[str, SourceKeyDescription]]]


### Base Class


class DataConnector:
    """Defines which data to pass to which component of the training pipeline.

    This data connector is used in the training / testing loop in order to
    provide input to the models / losses / evaluators and visualizers.

    It extracts the required data from a 'DictData' objects and passes it to
    the next component with the provided new key.
    """

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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def get_loss_input(
        self, prediction: DictData, data: DictData
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
        raise NotImplementedError()

    def get_callback_input(
        self,
        mode: str,
        prediction: DictData,
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

        Returns:
            dict[str, Tensor | DictStrArrayNested]: kwargs that are passed
                onto the callback.
        """
        raise NotImplementedError()


class StaticDataConnector(DataConnector):
    """DataConnector with static remapping of the keys."""

    def __init__(self, connections: DataConnectionInfo) -> None:
        """Creates  new DataConnector with the specified DataConnectionInfo.

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
