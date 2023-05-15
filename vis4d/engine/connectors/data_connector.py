"""Defines data structures for data connection."""
from __future__ import annotations

from torch import Tensor

from vis4d.common import DictStrArrNested
from vis4d.data.typing import DictData

from .util import SourceKeyDescription, get_inputs_for_pred_and_data


class DataConnector:
    """Defines which data to pass to which component of the training pipeline.

    This data connector is used in the training / testing loop in order to
    provide input to the models / losses.

    It extracts the required data from a 'DictData' objects and passes it to
    the next component with the provided new key.
    """

    def __init__(
        self,
        train: None | dict[str, str] = None,
        test: None | dict[str, str] = None,
        loss: None | dict[str, SourceKeyDescription] = None,
    ):
        """Initializes the data connector with static remapping of the keys.

        Args:
            train (dict[str, str], optional): Defines which kwargs to pass onto
                the model during training. Defaults to None.
            test (dict[str, str], optional): Defines which kwargs to pass onto
                the model during testing. Defaults to None.
            loss (dict[str, SourceKeyDescription], optional): Defines which
                kwargs to pass onto the loss. This field requres a key and a
                source for each entry since the loss takes data from the model
                output and data input. Defaults to None.


        Simple Example Configuration:

        >>> train = dict(images = "images", gt = "gt_images)
        >>> test = dict(images = "images")
        >>> loss = dict(box_pred = dict(key = "masks", source = "prediction"),
                        box_gt = dict(key = "masks", source = "data")
                    )
        >>> connector = DataConnector(train=train, test=test, loss=loss))
        """
        self.train = train
        self.test = test
        self.loss = loss

    def get_train_input(
        self, data: DictData
    ) -> dict[str, Tensor | DictStrArrNested]:
        """Returns the kwargs that are passed to the model for training.

        Args:
            data (DictData): The datadict (e.g. from the dataloader) which
                contains all data that was loaded.

        Returns:
            dict[str, Tensor | DictStrArrNested]: kwargs that are passed
                onto the model.
        """
        if self.train is None:
            return {}  # No data connections registered for training
        return {k: data[v] for k, v in self.train.items()}

    def get_test_input(
        self, data: DictData
    ) -> dict[str, Tensor | DictStrArrNested]:
        """Returns the kwargs that are passed to the model for testing.

        Args:
            data (DictData): The datadict (e.g. from the dataloader) which
                contains all data that was loaded.

        Returns:
            dict[str, Tensor | DictStrArrNested]: kwargs that are passed
                onto the model.
        """
        if self.test is None:
            return {}  # No data connections registered for testing
        return {k: data[v] for k, v in self.test.items()}

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
            dict[str, Tensor | DictStrArrNested]: kwargs that are passed
                onto the loss.
        """
        if self.loss is None:
            return {}  # No data connections registered for loss
        return get_inputs_for_pred_and_data(self.loss, prediction, data)
