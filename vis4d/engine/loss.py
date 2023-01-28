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


def _get_tensors_nested(  # type: ignore  # TODO: Move to util
    loss_dict: NestedLossesType, prefix=""
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


# def test_fn():
#     # Test case 1: Loss dict with tensors
#     loss_dict = {"loss1": torch.randn(1), "loss2": torch.randn(1)}
#     named_tensors = _get_tensors_nested(loss_dict)
#     assert len(named_tensors) == 2
#     assert isinstance(named_tensors[0][1], torch.Tensor)
#     assert isinstance(named_tensors[1][1], torch.Tensor)

#     # Test case 2: Loss dict with nested tensors
#     loss_dict = {"loss1": {"loss2": torch.randn(1)}}
#     named_tensors = _get_tensors_nested(loss_dict)
#     assert len(named_tensors) == 1
#     assert isinstance(named_tensors[0][1], torch.Tensor)

#     # Test case 3: Loss dict with nested dicts
#     loss_dict = {"loss1": {"loss2": {"loss3": torch.randn(1)}}}
#     named_tensors = _get_tensors_nested(loss_dict)
#     assert len(named_tensors) == 1
#     assert isinstance(named_tensors[0][1], torch.Tensor)

#     # Test case 4: Loss dict with non-tensor values
#     loss_dict = {"loss1": {"loss2": {"loss3": "not a tensor"}}}
#     try:
#         named_tensors = _get_tensors_nested(loss_dict)
#     except ValueError as e:
#         assert (
#             str(e)
#             == "Loss dict must only contain tensors or dicts.
#                   Found <class 'str'> at loss1.loss2.loss3."
#         )

#     # Test case 5: Loss dict with empty dict
#     loss_dict = {}
#     named_tensors = _get_tensors_nested(loss_dict)
#     assert len(named_tensors) == 0

#     # Test case 6: Loss dict with prefix
#     loss_dict = {"loss1": {"loss2": torch.randn(1)}}
#     prefix = "test"
#     named_tensors = _get_tensors_nested(loss_dict, prefix)
#     assert len(named_tensors) == 1
#     assert named_tensors[0][0] == "test.loss1.loss2"


class WeightedMultiLoss(nn.Module):
    """Weighted Multi Loss."""

    def __init__(self, losses: list[LossDefinition]) -> None:
        """Creates an instance of the class.

        Args:
            losses (list[nn.Module]): List of loss functions.
            weights (list[float]): List of weights.
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

        Args:
            **kwargs (Any): Arguments to pass to loss functions.

        Returns:
            torch.Tensor: Loss.
        """
        loss_dict: LossesType = {}
        for loss in self.losses:
            loss_values_as_dict: LossesType = {}
            name = loss["name"]
            # Save loss value
            loss_value = loss["loss"](
                **{
                    key_o: kwargs.get(key_i, None)  # TODO, maybe raise warning
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
                while key in loss_values_as_dict:
                    # TODO, show warning once
                    key = "__" + key

                loss_dict[key] = loss["weight"] * value

        return loss_dict
        # losses = [loss() for loss in self.losses]
        # return sum(
        #     [loss * weight for loss, weight in zip(losses, self.weights)]
        # )

    def __call__(self, **kwargs: Any) -> torch.Tensor:  # type: ignore
        """Type definition for call implementation."""
        return self._call_impl(**kwargs)


# class STH(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, test=0) -> torch.Tensor:
#         return {
#             "mutli": torch.tensor(1.0),
#             "nested": {"loss": torch.tensor(1.0)},
#         }
#         # return {"mutli": torch.tensor(1.0), "loss": torch.tensor(1.0)}


# if __name__ == "__main__":

#     print("hi")
#     l = Box3DUncertaintyLoss()
#     print(l)
#     import inspect

#     wm = WeightedMultiLoss(
#         [
#             LossDefinition(loss=STH(), weight=1),
#             LossDefinition(loss=STH(), weight=1),
#         ]
#     )
#     print(wm(test=1), "called")
#     print(inspect.signature(l.forward).parameters.keys())
#     print(inspect.signature(STH().forward).parameters.keys())
