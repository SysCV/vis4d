"""Basic data augmentation class."""
from typing import Any, Callable, List, Tuple, TypeVar

import torch

from vis4d.common import COMMON_KEYS, DictData, DictStrAny


# TODO move to commen
def get_dict_nested(dictionary: DictStrAny, keys: List[str]) -> Any:
    """Get value in nested dict."""
    for key in keys:
        if key not in dictionary:
            raise ValueError(f"Key {key} not in dictionary!")
        dictionary = dictionary[key]
    return dictionary


def set_dict_nested(
    dictionary: DictStrAny, keys: List[str], value: Any
) -> None:
    """Set value in nested dict."""
    for key in keys[:-1]:
        dictionary = dictionary.setdefault(key, {})
    dictionary[keys[-1]] = value


class Transform:
    """Decorator for transforms, which adds `in_keys` and `out_keys`.

    This decorator defines which keys are input to the transformation function
    and which keys are overwritten in the data dictionary by the output of
    the transformation.
    Nested keys in the data dictionary can be accessed via key.subkey1.subkey2

    Example:
        @Transform(in_keys=["image"], out_keys=["image"])
        def my_transform(option_a, option_b):
            def _transform(image):
                return do_transform(image)
            return _transform
    """

    def __init__(
        self,
        in_keys: Tuple[str, ...] = (COMMON_KEYS.images,),
        out_keys: Tuple[str, ...] = (COMMON_KEYS.images,),
        with_data: bool = False,
    ):
        """Init.

        Args:
            in_keys (List[str]): Input keys in the data dictionary. Nested keys are separated by '.', e.g. metadata.image_size.
            out_keys (List[str]): Output keys (possibly nested).
            with_data (bool, optional): Pass the full data dict as auxiliary input. Defaults to False.
        """
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.with_data = with_data

    def __call__(self, orig_get_transform_fn):
        """Wrap function with a handler for input / output keys."""

        def get_transform_fn(
            *args, in_keys=self.in_keys, out_keys=self.out_keys, **kwargs
        ):
            orig_transform_fn = orig_get_transform_fn(*args, **kwargs)

            def _transform_fn(data):
                in_data = [
                    get_dict_nested(data, key.split(".")) for key in in_keys
                ]
                # Optionally allow the function to get the full data dict as aux input.
                if self.with_data:
                    result = orig_transform_fn(*in_data, data=data)
                else:
                    result = orig_transform_fn(*in_data)
                if len(self.out_keys) == 1:
                    result = [result]
                for key, value in zip(out_keys, result):
                    set_dict_nested(data, key.split("."), value)
                return data

            return _transform_fn

        return get_transform_fn


class BatchTransform:
    """Decorator for batch transforms, which adds `in_keys` and `out_keys`."""

    def __init__(
        self,
        in_keys=(COMMON_KEYS.images,),
        out_keys=(COMMON_KEYS.images,),
        with_data: bool = False,
    ):
        """Init.

        Args:
            in_keys (List[str]): Input keys in the data dictionary. Nested keys are separated by '.', e.g. metadata.image_size.
            out_keys (List[str]): Output keys (possibly nested).
            with_data (bool, optional): Pass the full data dict as auxiliary input. Defaults to False.
        """
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.with_data = with_data

    def __call__(self, orig_get_transform_fn):
        """Wrap function with a handler for input / output keys."""

        def get_transform_fn(
            *args, in_keys=self.in_keys, out_keys=self.out_keys, **kwargs
        ):
            orig_transform_fn = orig_get_transform_fn(*args, **kwargs)

            def _transform_fn(batch):
                in_batch = []
                for key in in_keys:
                    key_data = []
                    for data in batch:
                        key_data.append(get_dict_nested(data, key.split(".")))
                    in_batch.append(key_data)
                # Optionally allow the function to get the full data dict as aux input.
                if self.with_data:
                    result = orig_transform_fn(*in_batch, data=batch)
                else:
                    result = orig_transform_fn(*in_batch)

                if len(self.out_keys) == 1:
                    result = [result]
                for key, values in zip(out_keys, result):
                    for data, value in zip(batch, values):
                        set_dict_nested(data, key.split("."), value)
                return batch

            return _transform_fn

        return get_transform_fn


TInput = TypeVar("TInput")


def compose(
    transforms: List[Callable[[TInput], TInput]]
) -> Callable[[TInput], TInput]:
    """Compose transformations."""

    def _preprocess_func(data: TInput) -> TInput:
        for op in transforms:
            data = op(data)
        return data

    return _preprocess_func


def random_apply(
    transforms: List[Callable[[DictData], DictData]], p: float = 0.5
):
    """Apply given transforms at random with given probability."""

    def _apply(data: DictData) -> DictData:
        data["transform_params"]["random_apply"] = False
        if torch.rand(1) < p:
            data["transform_params"]["random_apply"] = True
            for op in transforms:
                data = op(data)
        return data

    return _apply
