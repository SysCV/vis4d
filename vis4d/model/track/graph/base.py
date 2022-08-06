"""Tracking base class."""

import abc
from typing import List, Optional, Union, cast, overload

import torch
from torch import nn

from vis4d.struct import Losses


class BaseTrackGraph(nn.Module):
    """Base class for tracking graph optimization."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset track memory during inference."""
        raise NotImplementedError

    def forward(
        self,
        inputs,
        predictions,
        targets=None,
        **kwargs: Union[List[torch.Tensor], torch.Tensor],
    ):
        """Forward method. Decides between train / test logic."""
        if targets is not None:  # pragma: no cover
            return self.forward_train(inputs, predictions, targets, **kwargs)
        inputs = cast(InputSample, inputs)
        return self.forward_test(inputs, predictions, **kwargs)

    @abc.abstractmethod
    def forward_train(
        self,
        inputs,
        predictions,
        targets,
        **kwargs: List[torch.Tensor],
    ) -> Losses:
        """Process inputs, match detections with existing tracks."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self,
        inputs,
        predictions,
        **kwargs: torch.Tensor,
    ):
        """Process inputs, match detections with existing tracks."""
        raise NotImplementedError
