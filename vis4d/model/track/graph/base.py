"""Tracking base class."""

import abc
from typing import List, Optional, Union, cast, overload

import torch

from vis4d.common import Vis4DModule
from vis4d.struct import InputSample, LabelInstances, Losses


class BaseTrackGraph(Vis4DModule[LabelInstances, Losses]):
    """Base class for tracking graph optimization."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset track memory during inference."""
        raise NotImplementedError

    @overload  # type: ignore[override]
    def __call__(
        self,
        inputs: InputSample,
        predictions: LabelInstances,
        **kwargs: torch.Tensor,
    ) -> LabelInstances:  # noqa: D102
        ...

    @overload
    def __call__(
        self,
        inputs: List[InputSample],
        predictions: List[LabelInstances],
        targets: Optional[List[LabelInstances]],
        **kwargs: List[torch.Tensor],
    ) -> Losses:
        ...

    def __call__(
        self,
        inputs: Union[List[InputSample], InputSample],
        predictions: Union[List[LabelInstances], LabelInstances],
        targets: Optional[List[LabelInstances]] = None,
        **kwargs: Union[List[torch.Tensor], torch.Tensor],
    ) -> Union[LabelInstances, Losses]:
        """Forward method. Decides between train / test logic."""
        if targets is not None:  # pragma: no cover
            inputs = cast(List[InputSample], inputs)
            predictions = cast(List[LabelInstances], predictions)
            return self.forward_train(inputs, predictions, targets, **kwargs)
        inputs = cast(InputSample, inputs)
        predictions = cast(LabelInstances, predictions)
        return self.forward_test(inputs, predictions, **kwargs)

    @abc.abstractmethod
    def forward_train(
        self,
        inputs: List[InputSample],
        predictions: List[LabelInstances],
        targets: List[LabelInstances],
        **kwargs: List[torch.Tensor],
    ) -> Losses:
        """Process inputs, match detections with existing tracks."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self,
        inputs: InputSample,
        predictions: LabelInstances,
        **kwargs: torch.Tensor,
    ) -> LabelInstances:
        """Process inputs, match detections with existing tracks."""
        raise NotImplementedError
