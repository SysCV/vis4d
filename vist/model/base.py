"""Base class for VisT models."""

import abc
from typing import List, Tuple, Dict, Optional

import pytorch_lightning as pl
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from vist.common.registry import ABCRegistryHolder
from vist.struct import Boxes2D, InputSample, LossesType, ModelOutput


class BaseModelConfig(PydanticBaseModel, extra="allow"):
    """Config for default VisT tracker."""

    type: str = Field(...)
    category_mapping: Optional[Dict[str, int]] = None


class BaseModel(pl.LightningModule, metaclass=ABCRegistryHolder):
    """Base tracker class."""

    @abc.abstractmethod
    def training_step(
        self, batch_inputs: List[List[InputSample]]
    ) -> LossesType:
        """Forward pass during training stage.

        Args:
            batch_inputs: Model input. Batched, including possible reference
            views.

        Returns:
            LossesType: A dict of scalar loss tensors.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def testing_step(
        self, batch_inputs: List[InputSample]
    ) -> ModelOutput:
        """Forward pass during testing stage.

        Args:
            batch_inputs: Model input (batched).

        Returns:
            ModelOutput: Dict of LabelInstance results, e.g. tracking and
            separate models result.
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
    """Build VisT model.

    Note that it does not load any weights from ``cfg``.
    """
    registry = ABCRegistryHolder.get_registry(BaseModel)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseModel)
        return module
    raise NotImplementedError(f"Model {cfg.type} not found.")
