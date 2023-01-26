"""Basic data augmentation class."""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch

from vis4d.common.dict import get_dict_nested, set_dict_nested
from vis4d.common.prettyprint import PrettyRepMixin
from vis4d.common.typing import ArgsType
from vis4d.data.typing import DictData

AnyFunction = Callable  # type: ignore
TInput = TypeVar("TInput")  # pylint: disable=invalid-name


class Transform(PrettyRepMixin):
    """Base class for transforms.

    This class stores which `in_keys` are input to a transformation function
    and which `out_keys` are overwritten in the data dictionary by the output
    of this transformation.
    Nested keys in the data dictionary can be accessed via key.subkey1.subkey2
    If any of `in_keys` is 'data', the full data dictionary will be forwarded
    to the transformation.

    Example:
        >>> @Transform.register(in_keys=["images"], out_keys=["images"])
        >>> def my_transform(images):
        >>>     image = do_something(images)
        >>>     return image

    For the case of multi-sensor data, the sensors that the transform should be
    applied can be set via the 'sensors' attribute. By default, we assume
    a transformation is applied to all sensors.
    """

    _registered_in_keys: list[list[str]] = []
    _registered_out_keys: list[list[str]] = []
    _registered_funcs: list[AnyFunction] = []
    _remapped_keys: dict[str, str] = {}

    def __init__(
        self,
        sensors: None | list[str] | str = None,
        remap_in_keys: None | list[tuple[str, str]] | tuple[str, str] = None,
        remap_out_keys: None | list[tuple[str, str]] | tuple[str, str] = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            sensors (None | list[str] | str, optional): Specifies the sensors
                this transformation should be applied to. If None, it will be
                applied to all available sensors. Defaults to None.
            remap_in_keys (None | list[tuple[str, str]] | tuple[str, str],
                optional): Specifies one or multiple (if any) input keys of the
                data dictionary which should be remapeed to another key.
                Defaults to None.
            remap_out_keys (None | list[tuple[str, str]] | tuple[str, str],
                optional): Specifies one or multiple (if any) input keys of the
                data dictionary which should be remapeed to another key.
                Defaults to None.
        """
        assert isinstance(sensors, (list, str)) or sensors is None
        if isinstance(sensors, str):
            sensors = [sensors]
        self.sensors = sensors

        if remap_in_keys is not None:
            assert isinstance(remap_in_keys, (list, tuple))
            if isinstance(remap_in_keys, tuple):
                remap_in_keys = [remap_in_keys]
            for in_key, new_key in remap_in_keys:
                self.remap(in_key, new_key)

        if remap_out_keys is not None:
            assert isinstance(remap_out_keys, (list, tuple))
            if isinstance(remap_out_keys, tuple):
                remap_out_keys = [remap_out_keys]
            for out_key, new_key in remap_out_keys:
                self.remap(out_key, new_key)

    @classmethod
    def register(
        cls, in_keys: list[str] | str, out_keys: list[str] | str
    ) -> AnyFunction:
        if isinstance(in_keys, str):
            in_keys = [in_keys]
        if isinstance(out_keys, str):
            out_keys = [out_keys]

        def _register(func: AnyFunction) -> None:
            assert isinstance(in_keys, list)
            assert isinstance(out_keys, list)
            cls._registered_funcs.append(func)
            cls._registered_in_keys.append(in_keys)
            cls._registered_out_keys.append(out_keys)

        return _register

    @property
    def _param_in_keys(self) -> list[str]:
        raise NotImplementedError

    def _generate_parameters(
        self, *args: ArgsType, **kwargs: ArgsType
    ) -> None:
        raise NotImplementedError

    def _apply(
        self, data: DictData, generate_parameters: bool = True
    ) -> DictData:
        if generate_parameters:
            param_in_keys = [
                k if k not in self._remapped_keys else self._remapped_keys[k]
                for k in self._param_in_keys
            ]
            in_data = [
                get_dict_nested(data, key.split(".")) for key in param_in_keys
            ]
            self._generate_parameters(*in_data)

        for transform_func, in_keys, out_keys in zip(
            self._registered_funcs,
            self._registered_in_keys,
            self._registered_out_keys,
        ):
            try:
                in_data = [
                    get_dict_nested(data, key.split("."))
                    if key != "data"
                    else data
                    for key in in_keys
                ]
            except ValueError:
                # if a key does not exist in the input data, continue.
                # TODO we might want to raise a warning here, but just once...
                continue

            # Optionally allow the function to get the full data dict as
            # aux input.
            result = transform_func(*in_data, self)
            if len(out_keys) == 1:
                result = [result]
            for key, value in zip(out_keys, result):
                set_dict_nested(data, key.split("."), value)

        return data

    def __call__(
        self, data: DictData, generate_parameters: bool = True
    ) -> DictData:
        if self.sensors is not None:
            for sensor in self.sensors:
                data[sensor] = self._apply(data[sensor], generate_parameters)
            return data
        return self._apply(data)

    def remap(self, key: str, new_key: str) -> None:
        self._registered_in_keys = [
            [k if k != key else new_key for k in k_list]
            for k_list in self._registered_in_keys
        ]
        self._registered_out_keys = [
            [k if k != key else new_key for k in k_list]
            for k_list in self._registered_out_keys
        ]
        self._remapped_keys[key] = new_key


BatchTransform = None  # TODO


def compose(
    transforms: list[Callable[[TInput], TInput]]
) -> Callable[[TInput], TInput]:
    """Compose transformations.

    This function composes a given set of transformation functions into a
    single function.
    """

    def _preprocess_func(data: TInput) -> TInput:
        for op in transforms:
            data = op(data)
        return data

    return _preprocess_func


def random_apply(
    transforms: list[Callable[[TInput], TInput]], probability: float = 0.5
) -> Callable[[TInput], TInput]:
    """Apply given transforms at random with given probability.

    Args:
        transforms (list[Callable[[TInput], TInput]]): Transformations that
            are applied with a given probability.
        probability (float, optional): Probability to apply transformations.
            Defaults to 0.5.

    Returns:
        Callable[[TInput], TInput]]: The randomized transformations.
    """

    def _apply(data: DictData) -> DictData:
        if torch.rand(1) < probability:
            for op in transforms:
                data = op(data)
        return data

    return _apply
