"""Defines data structures for data connection."""
from __future__ import annotations

from copy import deepcopy
from typing import TypedDict

from torch import Tensor

from vis4d.common import DictStrArrNested

from vis4d.data.typing import DictData


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


class DataConnectionInfo(TypedDict):
    """Defines how and which entries to connect.

    This datastructure defines which kwargs to pass to the connected component.

    Attributes:
        train (dict[str, str]): Defines which kwargs to pass onto the model
            during training.
        test (dict[str, str]): Defines which kwargs to pass onto the model
            during testing.
        loss (dict[str, SourceKeyDescription]): Defines which kwargs to pass
            onto the loss. This field requres a key and a source for each entry
            since the loss takes data from the model output and data input.
        callbacks (dict[str, dict[str, SourceKeyDescription]], optional):
            Defines which kwargs to use for different callbacks. This data
            structure is slightly more complicated, since we could have
            different callbacks that might depend on model outputs and data
            from the dataloader.

    Simple Example Configuration:

    >>> train = dict(images = "images", gt = "gt_images)
    >>> test = dict(images = "images")
    >>> loss = dict(box_pred = dict(key = "masks", source = "prediction"),
                    box_gt = dict(key = "masks", source = "data")
                )
    >>> callbacks = dict(
            mask_visualizer = dict(
                images = dict(key = "images", source = "data"),
                masks = dict(key = "masks", source = "prediction"))
            ),
            semnatic_evaluator = dict(
                prediction = dict(key = "masks", source = "prediction"))
                groundtruth = dict(key = "masks", source = "data"))
            ),
        )
    >>> final_connection_dict = DataConnectionInfo(
            train =  train,
            test = test,
            loss = loss,
            callbacks = callbacks,
        )
    """

    train: dict[str, str]
    test: dict[str, str]
    loss: dict[str, SourceKeyDescription]
    callbacks: None | dict[str, dict[str, SourceKeyDescription]]


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

    # TODO: Maybe refactor this into a separate class for train / test / val
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
