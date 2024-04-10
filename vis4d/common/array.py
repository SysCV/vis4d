"""This module contains array utility functions."""

from __future__ import annotations

from typing import overload

import numpy as np
import torch

from vis4d.common.typing import (
    ArrayLike,
    NDArrayBool,
    NDArrayFloat,
    NDArrayInt,
    NDArrayNumber,
    NDArrayUInt,
    NumpyBool,
    NumpyFloat,
    NumpyInt,
    NumpyUInt,
)


@overload
def array_to_numpy(
    data: ArrayLike, n_dims: int | None, dtype: type[NumpyBool]
) -> NDArrayBool: ...


@overload
def array_to_numpy(
    data: ArrayLike, n_dims: int | None, dtype: type[NumpyFloat]
) -> NDArrayFloat: ...


@overload
def array_to_numpy(
    data: ArrayLike, n_dims: int | None, dtype: type[NumpyInt]
) -> NDArrayInt: ...


@overload
def array_to_numpy(
    data: ArrayLike, n_dims: int | None, dtype: type[NumpyUInt]
) -> NDArrayUInt: ...


@overload
def array_to_numpy(
    data: ArrayLike | None, n_dims: int | None, dtype: type[NumpyBool]
) -> NDArrayBool | None: ...


@overload
def array_to_numpy(
    data: ArrayLike | None, n_dims: int | None, dtype: type[NumpyFloat]
) -> NDArrayFloat | None: ...


@overload
def array_to_numpy(
    data: ArrayLike | None, n_dims: int | None, dtype: type[NumpyInt]
) -> NDArrayInt | None: ...


@overload
def array_to_numpy(
    data: ArrayLike | None, n_dims: int | None, dtype: type[NumpyUInt]
) -> NDArrayUInt | None: ...


@overload
def array_to_numpy(data: ArrayLike, n_dims: int | None) -> NDArrayNumber: ...


@overload
def array_to_numpy(data: None) -> None: ...


def array_to_numpy(
    data: ArrayLike | None,
    n_dims: int | None = None,
    dtype: (
        type[NumpyBool] | type[NumpyFloat] | type[NumpyInt] | type[NumpyUInt]
    ) = np.float32,
) -> NDArrayNumber | None:
    """Converts a given array like object to a numpy array.

    Helper function to convert an array like object to a numpy array.
    This functions converts torch.Tensors or Sequences to numpy arrays.

    If the argument is None, None will be returned.

    Examples:
    >>> convert_to_array([1,2,3])
    >>> # -> array([1,2,3])
    >>> convert_to_array(None)
    >>> # -> None
    >>> convert_to_array(torch.tensor([1,2,3]).cuda())
    >>> # -> array([1,2,3])
    >>> convert_to_array([1,2,3], n_dims = 2).shape
    >>> # -> [1, 3]

    Args:
        data (ArrayLike | None): ArrayLike object that should be converted
            to numpy.

        n_dims (int | None, optional): Target number of dimension of the array.
            If the provided array does not have this shape, it will be
            squeezed or exanded (from the left). If it still does not match,
            an error is raised.

        dtype (type[NumpyBool] | type[NumpyFloat] | type[NumpyInt] |
            type[NumpyUInt], optional): Target dtype of the array. Defaults to
            np.float32.

    Raises:
        ValueError: If the provied array like objects can not be converted
            with the target dimensions.

    Returns:
        NDArrayNumber | None: The converted numpy array or None if None was
            provided.
    """
    if data is None:
        return data

    if isinstance(data, np.ndarray):
        array = data
    elif isinstance(data, torch.Tensor):
        array = np.asarray(data.detach().cpu().numpy())
    else:
        array = np.asarray(data)

    if n_dims is not None:
        # Squeeze if needed
        for _ in range(len(array.shape) - n_dims):
            if array.shape[0] == 1:
                array = array.squeeze(0)
            elif array.shape[-1] == 1:
                array = array.squeeze(-1)

        # expand if needed
        for _ in range(n_dims - len(array.shape)):
            array = np.expand_dims(array, 0)

        if len(array.shape) != n_dims:
            raise ValueError(
                f"Failed to convert target array of shape {array.shape} to"
                f"have {n_dims} dimensions."
            )

    # hardcode next type check since mypy can not resolve this correctly
    typed_arr: NDArrayNumber = array.astype(dtype)  # type: ignore
    return typed_arr


@overload
def arrays_to_numpy(
    *args: ArrayLike, n_dims: int | None, dtype: type[NumpyBool]
) -> tuple[NDArrayBool, ...]: ...


@overload
def arrays_to_numpy(
    *args: ArrayLike, n_dims: int | None, dtype: type[NumpyFloat]
) -> tuple[NDArrayFloat, ...]: ...


@overload
def arrays_to_numpy(
    *args: ArrayLike, n_dims: int | None, dtype: type[NumpyInt]
) -> tuple[NDArrayInt, ...]: ...


@overload
def arrays_to_numpy(
    *args: ArrayLike, n_dims: int | None, dtype: type[NumpyUInt]
) -> tuple[NDArrayUInt, ...]: ...


def arrays_to_numpy(
    *args: ArrayLike | None,
    n_dims: int | None = None,
    dtype: (
        type[NumpyBool] | type[NumpyFloat] | type[NumpyInt] | type[NumpyUInt]
    ) = np.float32,
) -> tuple[NDArrayNumber | None, ...]:
    """Converts a given sequence of optional ArrayLike objects to numpy.

    Args:
        args (ArrayLike | None): Provided arguments.
        n_dims (int | None, optional): Target number of dimension of the array.
            If the provided array does not have this shape, it will be
            squeezed or exanded (from the left). If it still does not match,
            an error is Raised.
        dtype (type[NumpyBool] | type[NumpyFloat] | type[NumpyInt] |
            type[NumpyUInt], optional): Target dtype of the array. Defaults to
            np.float32.

    Raises:
        ValueError: If the provied array like objects can not be converted
            with the target dimensions.

    Returns:
        tuple[NDArrayNumber | None]: The converted arguments as numpy array.
    """
    # Ignore mypy check due to 'Not all union combinations were tried because
    # there are too many unions'
    return tuple(array_to_numpy(arg, n_dims, dtype) for arg in args)  # type: ignore # pylint: disable=line-too-long
