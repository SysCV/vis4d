"""Basic data augmentation class."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, TypeVar

import torch

from vis4d.common.dict import get_dict_nested, set_dict_nested
from vis4d.common.prettyprint import PrettyRepMixin
from vis4d.common.typing import ArgsType
from vis4d.data.typing import DictData

AnyFunction = Callable  # type: ignore
TInput = TypeVar("TInput")  # pylint: disable=invalid-name
TupleGetter = Any  # type: ignore
TransformParam = NamedTuple  # type: ignore


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
    _registered_param_keys: list[list[TupleGetter]] = []
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
        cls,
        in_keys: list[str] | str,
        out_keys: list[str] | str,
        param_keys: TupleGetter | list[TupleGetter] | None = None,
    ) -> AnyFunction:
        if isinstance(in_keys, str):
            in_keys = [in_keys]
        if isinstance(out_keys, str):
            out_keys = [out_keys]

        if param_keys is not None:
            if not isinstance(param_keys, list):
                param_keys = [param_keys]
        else:
            param_keys = []

        def _register(func: AnyFunction) -> None:
            assert isinstance(in_keys, list)
            assert isinstance(out_keys, list)
            cls._registered_funcs.append(func)
            cls._registered_in_keys.append(in_keys)
            cls._registered_out_keys.append(out_keys)
            cls._registered_param_keys.append(param_keys)

        return _register

    def parameter_in_keys(self) -> str | list[str]:
        raise NotImplementedError

    def generate_parameters(
        self, *args: ArgsType, **kwargs: ArgsType
    ) -> TransformParam:
        raise NotImplementedError

    def _get_obj_id(self) -> str:
        return str(id(self))

    @classmethod
    def _get_attr_from_tuple(
        cls, the_tuple: NamedTuple, getter: TupleGetter
    ) -> str:
        tuple_type = type(the_tuple)
        getters = [getattr(tuple_type, f) for f in tuple_type._fields]
        field_index = getters.index(getter)
        return the_tuple[field_index]

    def _get_param_from_dict(self, data: DictData) -> TransformParam | None:
        if "transforms" not in data.keys():
            return None
        elif self._get_obj_id() not in data["transforms"].keys():
            return None
        else:
            return data["transforms"][self._get_obj_id()]

    def _apply(self, data: DictData) -> DictData:
        parameters = self._get_param_from_dict(data)
        if parameters is None:
            param_in_keys = self.parameter_in_keys()
            if isinstance(param_in_keys, str):
                param_in_keys = [param_in_keys]
            param_in_keys = [
                k if k not in self._remapped_keys else self._remapped_keys[k]
                for k in param_in_keys
            ]
            in_data = [
                get_dict_nested(data, key.split(".")) for key in param_in_keys
            ]
            parameters = self.generate_parameters(*in_data)
            if not "transforms" in data.keys():
                data["transforms"] = {}
            data["transforms"][self._get_obj_id()] = parameters

        for transform_func, in_keys, out_keys, param_keys in zip(
            self._registered_funcs,
            self._registered_in_keys,
            self._registered_out_keys,
            self._registered_param_keys,
        ):
            try:
                # Optionally allow the function to get the full data dict as
                # aux input.
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

            in_params = [
                self._get_attr_from_tuple(parameters, getter)
                for getter in param_keys
            ]

            result = transform_func(*in_data, *in_params)
            if len(out_keys) == 1:
                result = [result]
            for key, value in zip(out_keys, result):
                set_dict_nested(data, key.split("."), value)

        return data

    def __call__(self, data: DictData) -> DictData:
        if self.sensors is not None:
            for sensor in self.sensors:
                data[sensor] = self._apply(data[sensor])
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
