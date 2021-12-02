"""Dense Head interface for Vis4D."""

import abc
from typing import Dict, Optional, Sequence, Union, overload

from pydantic import BaseModel, Field

from vis4d.common.module import Vis4DModule
from vis4d.common.registry import RegistryHolder
from vis4d.struct import (
    FeatureMaps,
    InputSample,
    LabelInstances,
    LossesType,
    TLabelInstance,
)


class BaseDenseHeadConfig(BaseModel, extra="allow"):
    """Base config for Dense head."""

    type: str = Field(...)
    category_mapping: Optional[Dict[str, int]] = None


class BaseDenseHead(Vis4DModule[LossesType, Sequence[TLabelInstance]]):
    """Base Dense head class."""

    @overload  # type: ignore[override]
    def forward(
        self,
        inputs: InputSample,
        features: Optional[FeatureMaps],
    ) -> Sequence[TLabelInstance]:  # noqa: D102
        ...

    @overload
    def forward(
        self,
        inputs: InputSample,
        features: Optional[FeatureMaps],
        targets: LabelInstances,
    ) -> LossesType:
        ...

    def forward(
        self,
        inputs: InputSample,
        features: Optional[FeatureMaps] = None,
        targets: Optional[LabelInstances] = None,
    ) -> Union[LossesType, Sequence[TLabelInstance]]:
        """Base Dense head forward.

        Args:
            inputs: Model Inputs, batched.
            features: Input feature maps.
            targets: Container with targets, e.g. Boxes2D / 3D, Masks, ...

        Returns:
            LossesType or Sequence[TLabelInstance]: In train mode, return
            losses. In test mode, return predictions.
        """
        if targets is not None:
            return self.forward_train(inputs, features, targets)
        return self.forward_test(inputs, features)

    @abc.abstractmethod
    def forward_train(
        self,
        inputs: InputSample,
        features: Optional[FeatureMaps],
        targets: LabelInstances,
    ) -> LossesType:
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            targets: Targets corresponding to InputSamples.

        Returns:
            LossesType: A dict of scalar loss tensors.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self,
        inputs: InputSample,
        features: Optional[FeatureMaps],
    ) -> Sequence[TLabelInstance]:
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.

        Returns:
            Sequence[TLabelInstance]: Prediction output.
        """
        raise NotImplementedError


def build_dense_head(
    cfg: BaseDenseHeadConfig,
) -> BaseDenseHead[TLabelInstance]:
    """Build a dense head from config."""
    registry = RegistryHolder.get_registry(BaseDenseHead)
    if cfg.type in registry:
        module = registry[cfg.type](cfg)
        assert isinstance(module, BaseDenseHead)
        return module
    raise NotImplementedError(f"DenseHead {cfg.type} not found.")
