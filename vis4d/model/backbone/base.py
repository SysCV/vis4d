"""Backbone interface for Vis4D."""
import abc
from typing import List, Optional, Tuple, Union

import torch

from vis4d.common.module import Vis4DModule, build_module
from vis4d.struct import FeatureMaps, InputSample, ModuleCfg

from .neck import BaseNeck


class BaseBackbone(Vis4DModule[FeatureMaps, FeatureMaps]):
    """Base Backbone class."""

    def __init__(
        self,
        pixel_mean: Tuple[float, float, float],
        pixel_std: Tuple[float, float, float],
        output_names: Optional[List[str]] = None,
        out_indices: Optional[List[int]] = None,
        neck: Optional[Union[BaseNeck, ModuleCfg]] = None,
    ) -> None:
        """Init BaseBackbone."""
        super().__init__()
        self.output_names = output_names
        self.out_indices = out_indices
        self.register_buffer(
            "pixel_mean",
            torch.tensor(pixel_mean).view(-1, 1, 1),
            False,
        )
        self.register_buffer(
            "pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False
        )

        if neck is not None:
            if isinstance(neck, dict):
                self.neck: Optional[BaseNeck] = build_module(
                    neck, bound=BaseNeck
                )
            else:
                self.neck = neck
        else:
            self.neck = None

    @abc.abstractmethod
    def preprocess_inputs(self, inputs: InputSample) -> InputSample:
        """Normalize the input images."""
        print(inputs.images.tensor.shape, self.pixel_mean.shape)
        inputs.images.tensor = (
            inputs.images.tensor - self.pixel_mean
        ) / self.pixel_std
        return inputs

    def get_outputs(self, outs: List[torch.Tensor]) -> FeatureMaps:
        """Get feature map dict."""
        if self.out_indices is not None:
            outs = [outs[ind] for ind in self.out_indices]
        if self.output_names is None:
            backbone_outs = {f"out{i}": v for i, v in enumerate(outs)}
        else:  # pragma: no cover
            assert len(self.output_names) == len(outs)
            backbone_outs = dict(zip(self.output_names, outs))
        return backbone_outs

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
