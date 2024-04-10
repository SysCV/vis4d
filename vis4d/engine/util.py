"""Run utilities."""

from __future__ import annotations

import dataclasses
from abc import ABC
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor

from vis4d.common.named_tuple import is_namedtuple

_BLOCKING_DEVICE_TYPES = ("cpu", "mps")


class TransferableDataType(ABC):
    """A custom type for data that can be moved to a torch device.

    Example:
        >>> isinstance(dict, TransferableDataType)
        False
        >>> isinstance(torch.rand(2, 3), TransferableDataType)
        True
        >>> class CustomObject:
        ...     def __init__(self):
        ...         self.x = torch.rand(2, 2)
        ...     def to(self, device):
        ...         self.x = self.x.to(device)
        ...         return self
        >>> isinstance(CustomObject(), TransferableDataType)
        True
    """

    @classmethod
    def __subclasshook__(cls, subclass: Any) -> bool | Any:  # type: ignore
        """Subclass hook."""
        if cls is TransferableDataType:
            to = getattr(subclass, "to", None)
            return callable(to)
        return NotImplemented  # pragma: no cover


def is_dataclass_instance(obj: object) -> bool:
    """Check if obj is dataclass instance.

    https://docs.python.org/3/library/dataclasses.html#module-level-decorators-classes-and-functions
    """
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def apply_to_collection(  # type: ignore
    data: Any,
    dtype: type | Any | tuple[type | Any],
    function: Callable[[Any], Any],
    *args: Any,
    wrong_dtype: None | type | tuple[type, ...] = None,
    include_none: bool = True,
    **kwargs: Any,
) -> Any:
    """Recursively applies a function to all elements of a certain dtype.

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of
            ``function``)
        wrong_dtype: the given function won't be applied if this type is
            specified and the given collections is of the ``wrong_dtype`` even
            if it is of type ``dtype``
        include_none: Whether to include an element if the output of
            ``function`` is ``None``.
        **kwargs: keyword arguments (will be forwarded to calls of
            ``function``)

    Raises:
        ValueError: If frozen dataclass is passed to `apply_to_collection`.

    Returns:
        The resulting collection
    """
    # Breaking condition
    if isinstance(data, dtype) and (
        wrong_dtype is None or not isinstance(data, wrong_dtype)
    ):
        return function(data, *args, **kwargs)

    elem_type = type(data)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        out = []
        for k, v in data.items():
            v = apply_to_collection(
                v,
                dtype,
                function,
                *args,
                wrong_dtype=wrong_dtype,
                include_none=include_none,
                **kwargs,
            )
            if include_none or v is not None:
                out.append((k, v))
        if isinstance(data, defaultdict):
            return elem_type(data.default_factory, OrderedDict(out))
        return elem_type(OrderedDict(out))

    is_namedtuple_ = is_namedtuple(data)
    is_sequence = isinstance(data, Sequence) and not isinstance(data, str)
    if is_namedtuple_ or is_sequence:
        out = []
        for d in data:
            v = apply_to_collection(
                d,
                dtype,
                function,
                *args,
                wrong_dtype=wrong_dtype,
                include_none=include_none,
                **kwargs,
            )
            if include_none or v is not None:
                out.append(v)
        return elem_type(*out) if is_namedtuple_ else elem_type(out)

    if is_dataclass_instance(data):
        # make a deepcopy of the data,
        # but do not deepcopy mapped fields since the computation would
        # be wasted on values that likely get immediately overwritten
        fields = {}
        memo = {}
        for field in dataclasses.fields(data):
            field_value = getattr(data, field.name)
            fields[field.name] = (field_value, field.init)
            memo[id(field_value)] = field_value
        result = deepcopy(data, memo=memo)
        # apply function to each field
        for field_name, (field_value, field_init) in fields.items():
            v = None
            if field_init:
                v = apply_to_collection(
                    field_value,
                    dtype,
                    function,
                    *args,
                    wrong_dtype=wrong_dtype,
                    include_none=include_none,
                    **kwargs,
                )
            if not field_init or (
                not include_none and v is None
            ):  # retain old value
                v = getattr(data, field_name)
            try:
                setattr(result, field_name, v)
            except dataclasses.FrozenInstanceError as e:
                raise ValueError(
                    "A frozen dataclass was passed to `apply_to_collection` "
                    "but this is not allowed."
                ) from e
        return result

    # data is neither of dtype, nor a collection
    return data


def move_data_to_device(  # type: ignore
    batch: Any,
    device: torch.device | str | int,
    convert_to_numpy: bool = False,
) -> Any:
    """Transfers a collection of data to the given device.

    Any object that defines a method ``to(device)`` will be moved and all other
    objects in the collection will be left untouched.

    This implementation is modified from
    https://github.com/Lightning-AI/lightning

    Args:
        batch: A tensor or collection of tensors or anything that has a method
            ``.to(...)``. See :func:`apply_to_collection` for a list of
            supported collection types.
        device: The device to which the data should be moved.
        convert_to_numpy: Whether to convert from tensor to numpy array.

    Return:
        The same collection but with all contained tensors residing on the new
            device.
    """
    if isinstance(device, str):
        device = torch.device(device)

    def batch_to(data: Any) -> Any:  # type: ignore[misc]
        kwargs = {}
        # Don't issue non-blocking transfers to CPU
        # Same with MPS due to a race condition bug:
        # https://github.com/pytorch/pytorch/issues/83015
        if (
            isinstance(data, Tensor)
            and isinstance(device, torch.device)
            and device.type not in _BLOCKING_DEVICE_TYPES
        ):
            kwargs["non_blocking"] = True
        data_output = data.to(device, **kwargs)
        if data_output is not None:
            if convert_to_numpy:
                data_output = data_output.numpy()
            return data_output
        # user wrongly implemented the `TransferableDataType` and forgot to
        # return `self`.
        return data

    return apply_to_collection(
        batch, dtype=TransferableDataType, function=batch_to
    )
