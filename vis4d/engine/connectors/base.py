"""Base data connector to define data structures for data connection."""
from __future__ import annotations

from torch import Tensor

from vis4d.common.typing import DictStrArrNested
from vis4d.data.typing import DictData

from .util import SourceKeyDescription, get_inputs_for_pred_and_data


class DataConnector:
    """Defines which data to pass to which component.

    It extracts the required data from a 'DictData' objects and passes it to
    the next component with the provided new key.
    """

    def __init__(self, key_mapping: dict[str, str]):
        """Initializes the data connector with static remapping of the keys.

        Args:
            key_mapping (dict[str, str] | dict[str, SourceKeyDescription]):
                Defines which kwargs to pass onto the module.

        Simple Example Configuration:

        >>> train = dict(images = "images", gt = "gt_images)
        >>> train_data_connector = DataConnector(train)
        >>> test = dict(images = "images")
        >>> test_data_connector = DataConnector(test)
        """
        self.key_mapping = key_mapping

    def __call__(  # pytlint: disable=arguments-differ
        self, data: DictData
    ) -> dict[str, Tensor | DictStrArrNested]:
        """Returns the kwargs that are passed to the module.

        Args:
            data (DictData): The datadict (e.g. from the dataloader) which
                contains all data that was loaded.

        Returns:
            dict[str, Tensor | DictStrArrNested]: kwargs that are passed
                onto the model.
        """
        return {k: data[v] for k, v in self.key_mapping.items()}


class LossConnector:
    """Defines which data to pass to loss module of the training pipeline.

    It extracts the required data from prediciton and data and passes it to
    the next component with the provided new key.
    """

    def __init__(self, key_mapping: dict[str, SourceKeyDescription]) -> None:
        """Initializes the data connector with static remapping of the keys."""
        self.key_mapping = key_mapping

    def __call__(
        self, prediction: DictData, data: DictData
    ) -> dict[str, Tensor | DictStrArrNested]:
        """Returns the kwargs that are passed to the loss during training.

        Args:
            prediction (DictData): The datadict (e.g. output from model) which
                contains all the model outputs.
            data (DictData): The datadict (e.g. from the dataloader) which
                contains all data that was loaded.

        Returns:
            dict[str, Tensor | DictStrArrNested]: kwargs that are passed
                onto the loss.
        """
        return get_inputs_for_pred_and_data(self.key_mapping, prediction, data)


class CallbackConnector:
    """Data connector for the callback.

    It extracts the required data from prediciton and datas and passes it to
    the next component with the provided new key.
    """

    def __init__(self, key_mapping: dict[str, SourceKeyDescription]) -> None:
        """Initializes the data connector with static remapping of the keys."""
        self.key_mapping = key_mapping

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
        return get_inputs_for_pred_and_data(self.key_mapping, prediction, data)
