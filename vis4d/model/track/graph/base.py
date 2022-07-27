"""Tracking base class."""

import abc
from typing import List, Optional, Union, cast

import torch
from torch import nn

from vis4d.struct import InputSample, LabelInstances, LossesType


class BaseTrackGraph(nn.Module):
    """Base class for tracking graph optimization."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset track memory during inference."""
        raise NotImplementedError

    def forward(
        self,
        inputs: Union[List[InputSample], InputSample],
        predictions: Union[List[LabelInstances], LabelInstances],
        targets: Optional[List[LabelInstances]] = None,
        **kwargs: Union[List[torch.Tensor], torch.Tensor],
    ) -> Union[LabelInstances, LossesType]:
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
    ) -> LossesType:
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
