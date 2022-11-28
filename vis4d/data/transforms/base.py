"""Basic data augmentation class."""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch

from vis4d.common.dict import get_dict_nested, set_dict_nested
from vis4d.data.const import CommonKeys
from vis4d.data.typing import DictData


class Transform:
    """Decorator for transforms, which adds `in_keys` and `out_keys`.

    This decorator defines which keys are input to the transformation function
    and which keys are overwritten in the data dictionary by the output of
    the transformation.
    Nested keys in the data dictionary can be accessed via key.subkey1.subkey2

    Example:
        >>> @Transform(in_keys=["image"], out_keys=["image"])
        >>> def my_transform(option_a, option_b):
        >>>     def _transform(image):
        >>>         return do_transform(image)
        >>>     return _transform

    For the case of multi-sensor data, the sensors that the transform should be
    applied can be set via the 'sensors' attribute. By default, we assume
    single sensor data (DictData).
    """

    def __init__(
        self,
        in_keys: tuple[str, ...] = (CommonKeys.images,),
        out_keys: tuple[str, ...] = (CommonKeys.images,),
        sensors: None | tuple[str, ...] = None,
        with_data: bool = False,
    ):
        """Init.

        Args:
            in_keys (tuple[str, ...], optional): Input keys in the data
                dictionary. Nested keys are separated by '.', e.g.,
                metadata.image_size. Defaults to (COMMON_KEYS.images,).
            out_keys (tuple[str, ...], optional): Output keys (possibly
                nested). Defaults to (COMMON_KEYS.images,).
            sensors (None | tuple[str, ...], optional): This field indicates
                which sensors the transform should be applied to. Defaults to
                None.
            with_data (bool, optional): Pass the full data dict as auxiliary
                input. Defaults to False.
        """
        assert isinstance(in_keys, (list, tuple))
        assert isinstance(out_keys, (list, tuple))
        assert isinstance(sensors, (list, tuple)) or sensors is None
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.sensors = sensors
        self.with_data = with_data

    def __call__(self, orig_get_transform_fn):
        """Wrap function with a handler for input / output keys."""

        def get_transform_fn(
            *args,
            in_keys=self.in_keys,
            out_keys=self.out_keys,
            sensors=self.sensors,
            **kwargs,
        ):
            orig_transform_fn = orig_get_transform_fn(*args, **kwargs)

            def _transform_fn(data):  # TODO multi sensor error msg improve
                in_data = [
                    get_dict_nested(data, key.split(".")) for key in in_keys
                ]
                # Optionally allow the function to get the full data dict as
                # aux input.
                if self.with_data:
                    result = orig_transform_fn(*in_data, data=data)
                else:
                    result = orig_transform_fn(*in_data)
                if len(out_keys) == 1:
                    result = [result]
                for key, value in zip(out_keys, result):
                    set_dict_nested(data, key.split("."), value)
                return data

            if sensors is not None:

                def _multi_sensor_transform_fn(data):
                    for sensor in sensors:
                        data[sensor] = _transform_fn(data[sensor])
                    return data

                return _multi_sensor_transform_fn
            return _transform_fn

        return get_transform_fn


class BatchTransform:
    """Decorator for batch transforms, which adds `in_keys` and `out_keys`."""

    def __init__(
        self,
        in_keys=(CommonKeys.images,),
        out_keys=(CommonKeys.images,),
        sensors: None | tuple[str, ...] = None,
        with_data: bool = False,
    ):
        """Init.

        Args:
            in_keys (tuple[str, ...], optional): Input keys in the data
                dictionary. Nested keys are separated by '.', e.g.,
                metadata.image_size. Defaults to (COMMON_KEYS.images,).
            out_keys (tuple[str, ...], optional): Output keys (possibly
                nested). Defaults to (COMMON_KEYS.images,).
            sensors (None | tuple[str, ...], optional): This field indicates
                which sensors the transform should be applied to. Defaults to
                None.
            with_data (bool, optional): Pass the full data dict as auxiliary
                input. Defaults to False.
        """
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.sensors = sensors
        self.with_data = with_data

    def __call__(self, orig_get_transform_fn):
        """Wrap function with a handler for input / output keys."""

        def get_transform_fn(
            *args,
            in_keys=self.in_keys,
            out_keys=self.out_keys,
            sensors=self.sensors,
            **kwargs,
        ):
            orig_transform_fn = orig_get_transform_fn(*args, **kwargs)

            def _transform_fn(batch):
                in_batch = []
                for key in in_keys:
                    key_data = []
                    for data in batch:
                        key_data.append(get_dict_nested(data, key.split(".")))
                    in_batch.append(key_data)
                # Optionally allow the function to get the full data dict as
                # aux input.
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

            if sensors is not None:

                def _multi_sensor_transform_fn(batch):
                    for sensor in sensors:
                        batch_sensor = _transform_fn(
                            [d[sensor] for d in batch]
                        )
                        for i, d in enumerate(batch_sensor):
                            batch[i][sensor] = d
                    return batch

                return _multi_sensor_transform_fn
            return _transform_fn

        return get_transform_fn


TInput = TypeVar("TInput")


def compose(
    transforms: list[Callable[[TInput], TInput]]
) -> Callable[[TInput], TInput]:
    """Compose transformations."""

    def _preprocess_func(data: TInput) -> TInput:
        for op in transforms:
            data = op(data)
        return data

    return _preprocess_func


def random_apply(
    transforms: list[Callable[[DictData], DictData]], p: float = 0.5
):
    """Apply given transforms at random with given probability."""

    def _apply(data: DictData) -> DictData:
        if torch.rand(1) < p:
            for op in transforms:
                data = op(data)
        return data

    return _apply
