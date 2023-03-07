"""Loss implementations to be used with the CLI.

This file provides a set of loss implementations that can be used with the CLI.

Currently, it implements the following losses:
    WeightedMultiLoss: A loss that combines multiple losses with weights.
"""
from __future__ import annotations

import inspect
from typing import Any, TypedDict, Union

import torch
from torch import Tensor, nn
from typing_extensions import NotRequired

from vis4d.common.typing import LossesType
from vis4d.engine.util import is_namedtuple
from vis4d.op.loss.base import Loss


class LossDefinition(TypedDict):
    """Loss definition."""

    loss: Loss | nn.Module
    weight: float
    name: NotRequired[str]
    in_keys: NotRequired[dict[str, str]]


NestedLossesType = Union[dict[str, "NestedLossesType"], LossesType]


def _get_tensors_nested(
    loss_dict: NestedLossesType, prefix: str = ""
) -> list[tuple[str, Tensor]]:
    """Get tensors from loss dict.

    Args:
        loss_dict (LossesType): Loss dict.
        prefix (str, optional): Prefix to add to keys. Defaults to "".

    Returns:
        list[torch.Tensor]: List of tensors.

    Raises:
        ValueError: If loss dict contains non-tensor or dict values.
    """
    named_tensors: list[tuple[str, Tensor]] = []
    for key in loss_dict:
        value = loss_dict[key]

        if isinstance(value, torch.Tensor):
            named_tensors.append((prefix + key, value))
        elif isinstance(value, dict):
            named_tensors.extend(
                _get_tensors_nested(value, prefix + key + ".")
            )
        else:
            raise ValueError(
                f"Loss dict must only contain tensors or dicts. "
                f"Found {type(loss_dict[key])} at {prefix + key}."
            )
    return named_tensors


class WeightedMultiLoss(nn.Module):
    """Loss that combines multiple losses with weights.

    This loss combines multiple losses with weights. The loss values are
    weighted by the corresponding weight and returned as a dictionary.
    """

    def __init__(self, losses: list[LossDefinition]) -> None:
        """Creates an instance of the class.

        By default, each loss will be called with arguments matching the
        kwargs of the loss function. This behavior can be changed by
        providing a mapping in in_keys for each loss definition.

        Args:
            losses (list[nn.Module]): List of loss defintions.
            weights (list[float]): List of weights.

        Example:
            >>> loss = WeightedMultiLoss(
            >>>     [
            >>>         {"loss": nn.MSELoss(), "weight": 1.0},
            >>>         {"loss": nn.L1Loss(), "weight": 0.5},
            >>>     ]
            >>> )
        """
        super().__init__()
        self.losses: list[LossDefinition] = []

        for loss in losses:
            assert "loss" in loss
            assert "weight" in loss

            if "name" not in loss:
                loss["name"] = loss["loss"].__class__.__name__
            if "in_keys" not in loss:
                loss["in_keys"] = {}
                for k in inspect.signature(
                    loss["loss"].forward
                ).parameters.keys():
                    loss["in_keys"][k] = k

            if "args" in loss["in_keys"] or "kwargs" in loss["in_keys"]:
                raise ValueError(
                    "Loss functions must not have args or kwargs as "
                    "parameters. Please use explicit parameters or provide"
                    "the correspondinig key mapping in in_keys."
                )
            self.losses.append(loss)

    def forward(self, **kwargs: Any) -> LossesType:  # type: ignore
        """Forward of loss function.

        This function will call all loss functions and return a dictionary
        containing the loss values. The loss values are weighted by the
        corresponding weight.

        If two losses have the same name, the name will be appended with
        two underscores.

        Args:
            **kwargs (Any): Arguments to pass to loss functions.

        Returns:
            LossesType: The loss values.
        """
        loss_dict: LossesType = {}
        for loss in self.losses:
            loss_values_as_dict: LossesType = {}
            name = loss["name"]
            # Save loss value
            loss_value = loss["loss"](
                **{
                    key_o: kwargs.get(key_i, None)
                    for key_o, key_i in loss["in_keys"].items()
                }
            )

            # Convert loss value to one level dict.
            if isinstance(loss_value, torch.Tensor):
                # Loss returned a simple tensor
                loss_values_as_dict[name] = loss_value
            elif isinstance(loss_value, dict):
                # Loss returned a dictionary.
                for loss_name, loss_value in _get_tensors_nested(
                    loss_value, name + "."
                ):
                    loss_values_as_dict[loss_name] = loss_value
            elif is_namedtuple(loss_value):
                # Loss returned a named tuple.
                for loss_name, loss_value in zip(
                    loss_value._fields, loss_value
                ):
                    loss_values_as_dict[name + "." + loss_name] = loss_value
            # Assign values
            for key, value in loss_values_as_dict.items():
                while key in loss_dict:
                    key = "__" + key

                loss_dict[key] = loss["weight"] * value

        return loss_dict
