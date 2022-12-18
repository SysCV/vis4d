"""Array utility functions."""
from __future__ import annotations

from typing import overload

import numpy as np
import torch

from vis4d.common.typing import (
    ArrayLike,
    ArrayLikeBool,
    ArrayLikeInt,
    NDArrayBool,
    NDArrayInt,
    NDArrayNumber,
)


@overload
def convert_to_array(data: ArrayLikeBool, n_dims: int | None) -> NDArrayBool:
    ...


@overload
def convert_to_array(data: ArrayLikeInt, n_dims: int | None) -> NDArrayInt:
    ...


@overload
def convert_to_array(data: ArrayLike, n_dims: int | None) -> NDArrayNumber:
    ...


@overload
def convert_to_array(data: None, n_dims: int | None) -> None:
    ...


def convert_to_array(
    data: ArrayLike | None, n_dims: int | None = None
) -> NDArrayNumber | None:
    """Converts a given array like object to a numpy array.

    Helper function to convert an array like object to a numpy array.
    This functions converts torch.Tensors or Sequences to numpy arrays.

    If the argument is None, None will be returned.

    Example:
    >>> convert_to_array([1,2,3])
    >>> # -> array[1,2,3]
    >>> convert_to_array(None)
    >>> # -> None
    >>> convert_to_array(torch.tensor([1,2,3]).cuda())
    >>> # -> array[1,2,3]
    >>> convert_to_array([1,2,3], n_dims = 2).shape
    >>> # -> [1, 3]

    Args:
        data (ArrayLike | None): ArrayLike object that should be converted
            to numpy.

        n_dims (int | None, optional): Target number of dimension of the array.
            If the provided array does not have this shape, it will be
            squeezed or exanded (from the left). If it still does not match,
            an error is Raised.

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

    return array


def convert_to_arrays(
    *args: ArrayLike | None, n_dims: int | None = None
) -> tuple[NDArrayNumber | None, ...]:
    """Converts a given sequence of optional ArrayLike objects to numpy.

    Args:
        args (ArrayLike | None): Provided arguments.
        n_dims (int | None, optional): Target number of dimension of the array.
            If the provided array does not have this shape, it will be
            squeezed or exanded (from the left). If it still does not match,
            an error is Raised.

    Raises:
        ValueError: If the provied array like objects can not be converted
            with the target dimensions.

    Returns:
        tuple[NDArrayNumber | None]: The converted arguments as numpy array.
    """
    return tuple(convert_to_array(arg, n_dims) for arg in args)
