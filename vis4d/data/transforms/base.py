"""Basic data augmentation class."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import torch

from vis4d.common.dict import get_dict_nested, set_dict_nested
from vis4d.data.typing import DictData

TFunctor = TypeVar("TFunctor", bound=Callable)  # type: ignore
TransformFunction = Callable[[DictData], DictData]
TInput = TypeVar("TInput")  # pylint: disable=invalid-name


class Transform:
    """Transforms Decorator.

    This class stores which `in_keys` are input to a transformation function
    and which `out_keys` are overwritten in the data dictionary by the output
    of this transformation.
    Nested keys in the data dictionary can be accessed via key.subkey1.subkey2
    If any of `in_keys` is 'data', the full data dictionary will be forwarded
    to the transformation.
    For the case of multi-sensor data, the sensors that the transform should be
    applied can be set via the 'sensors' attribute. By default, we assume
    a transformation is applied to all sensors.
    This class will add a 'apply_to_data' method to a given Functor which is
    used to call it on a DictData object.

    Example:
        >>> @Transform(in_keys="images", out_keys="images")
        >>> class MyTransform:
        >>>     def __call__(images):
        >>>         image = do_something(images)
        >>>         return image
        >>> my_transform = MyTransform()
        >>> data = my_transform.apply_to_data(data)
    """

    def __init__(
        self,
        in_keys: Sequence[str] | str,
        out_keys: Sequence[str] | str,
        sensors: Sequence[str] | str | None = None,
    ) -> None:
        """Creates an instance of Transform.

        Args:
            in_keys (None | list[tuple[str, str]] | tuple[str, str],
                optional): Specifies one or multiple (if any) input keys of the
                data dictionary which should be remapeed to another key.
                Defaults to None.
            out_keys (None | list[tuple[str, str]] | tuple[str, str],
                optional): Specifies one or multiple (if any) input keys of the
                data dictionary which should be remapeed to another key.
                Defaults to None.
            sensors (None | list[str] | str, optional): Specifies the sensors
                this transformation should be applied to. If None, it will be
                applied to all available sensors. Defaults to None.
        """
        if isinstance(in_keys, str):
            in_keys = [in_keys]
        self.in_keys = in_keys

        if isinstance(out_keys, str):
            out_keys = [out_keys]
        self.out_keys = out_keys

        if isinstance(sensors, str):
            sensors = [sensors]
        self.sensors = sensors

    def __call__(self, transform: TFunctor) -> TFunctor:
        """Add in_keys / out_keys / sensors / apply_to_data attributes.

        Args:
            transform (TFunctor): A given Functor.

        Returns:
            TFunctor: The decorated Functor.
        """
        original_init = transform.__init__

        def apply_to_data(
            self_,
            input_data: DictData,
            in_keys: Sequence[str] = self.in_keys,
            out_keys: Sequence[str] = self.out_keys,
            sensors: Sequence[str] | None = self.sensors,
        ) -> DictData:
            """Wrap function with a handler for input / output keys.

            We use the specified in_keys in order to extract the positional
            input arguments of a function from the data dictionary, and the
            out_keys to replace the corresponding values in the output
            dictionary.
            """

            def _transform_fn(data: DictData) -> DictData:
                in_data = []
                for key in in_keys:
                    try:
                        # Optionally allow the function to get the full data
                        # dict as aux input.
                        in_data += [
                            get_dict_nested(data, key.split("."))
                            if key != "data"
                            else data
                        ]
                    except ValueError:
                        # if a key does not exist in the input data, do not
                        # apply the transformation.
                        # TODO might need to raise a warning
                        return data

                result = self_(*in_data)
                if len(out_keys) == 1:
                    result = [result]
                for key, value in zip(out_keys, result):
                    set_dict_nested(data, key.split("."), value)
                return data

            if sensors is not None:
                for sensor in sensors:
                    input_data[sensor] = _transform_fn(input_data[sensor])
            else:
                input_data = _transform_fn(input_data)
            return input_data

        def init(*args, **kwargs):
            self_ = args[0]
            original_init(*args, **kwargs)
            self_.in_keys = self.in_keys
            self_.out_keys = self.out_keys
            self_.sensors = self.sensors
            self_.apply_to_data = lambda *args, **kwargs: apply_to_data(
                self_, *args, **kwargs
            )

        transform.__init__ = init
        return transform


BatchTransform = None  # TODO


def compose(transforms: list[TFunctor]) -> TransformFunction:
    """Compose transformations.

    This function composes a given set of transformation functions, i.e. any
    function decorated with the Transform functor, into a single transform.
    """

    def _preprocess_func(data: DictData) -> DictData:
        for op in transforms:
            data = op.apply_to_data(data)
        return data

    return _preprocess_func


def random_apply(
    transforms: list[TFunctor], probability: float = 0.5
) -> TransformFunction:
    """Apply given transforms at random with given probability.

    Args:
        transforms (list[TFunctor]): Transformations that
            are applied with a given probability.
        probability (float, optional): Probability to apply transformations.
            Defaults to 0.5.

    Returns:
        TransformFunction: The randomized transformations.
    """

    def _apply(data: DictData) -> DictData:
        if torch.rand(1) < probability:
            for op in transforms:
                data = op.apply_to_data(data)
        return data

    return _apply
