"""Backbone interface for Vis4D."""

import abc
from typing import Optional, Union

from vis4d.common.module import Vis4DModule, build_module
from vis4d.struct import FeatureMaps, InputSample, ModuleCfg

from .neck import BaseNeck


class BaseBackbone(Vis4DModule[FeatureMaps, FeatureMaps]):
    """Base Backbone class."""

    def __init__(self, neck: Optional[Union[BaseNeck, ModuleCfg]]) -> None:
        """Init BaseBackbone."""
        super().__init__()
        if isinstance(neck, dict):
            self.neck: Optional[BaseNeck] = build_module(neck, bound=BaseNeck)
        else:
            self.neck = neck

    @abc.abstractmethod
    def preprocess_inputs(self, inputs: InputSample) -> InputSample:
        """Normalize the input images."""
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(  # type: ignore[override]
        self,
        inputs: InputSample,
    ) -> FeatureMaps:
        """Base Backbone forward.

        Args:
            inputs: Model Inputs, batched.

        Returns:
            FeatureMaps: Dictionary of output feature maps.
        """
        raise NotImplementedError
