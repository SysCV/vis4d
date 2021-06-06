"""Base class for openMT models."""

import abc
from typing import Dict, List, Tuple, Union

import torch
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from openmt.common.registry import RegistryHolder
from openmt.struct import Boxes2D, InputSample, LossesType, ModelOutput


class BaseModelConfig(PydanticBaseModel, extra="allow"):
    """Config for default openMT tracker."""

    type: str = Field(...)


class BaseModel(torch.nn.Module, metaclass=RegistryHolder):  # type: ignore
    """Base tracker class."""

    def forward(
        self, batch_inputs: List[List[InputSample]]
    ) -> Union[LossesType, ModelOutput]:
        """Model forward function."""
        if self.training:
            return self.forward_train(batch_inputs)
        inputs = [inp[0] for inp in batch_inputs]  # no ref views during test
        return self.forward_test(inputs)

    @abc.abstractmethod
    def forward_train(
        self, batch_inputs: List[List[InputSample]]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass during training stage.

        Args:
            batch_inputs: Model input. Batched, including possible reference
            views.

        Returns:
            Dict: A dict of loss tensors.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
        """Forward pass during testing stage.

        Args:
            batch_inputs: Model input (batched).

        Returns:
            ModelOutput: Dict of LabelInstance results, e.g. tracking and
            separate detection result.
        """
        raise NotImplementedError

    @staticmethod
    def postprocess(
        original_wh: Tuple[int, int],
        output_wh: Tuple[int, int],
        detections: Boxes2D,
    ) -> None:
        """Postprocess results."""
        scale_factor = (
            original_wh[0] / output_wh[0],
            original_wh[1] / output_wh[1],
        )
        detections.scale(scale_factor)
        detections.clip(original_wh)


def build_model(cfg: BaseModelConfig) -> BaseModel:
    """Build openMT model.

    Note that it does not load any weights from ``cfg``.
    """
    assert cfg is not None
    registry = RegistryHolder.get_registry(__package__)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseModel)
        return module
    raise NotImplementedError(f"Model {cfg.type} not found.")
