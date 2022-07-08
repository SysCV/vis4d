"""Tracking base class."""

import abc
from typing import List, Optional, Union, cast, overload

import torch
from torch import nn

from vis4d.struct import InputSample, Losses


class BaseTrackGraph(nn.Module):
    """Base class for tracking graph optimization."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset track memory during inference."""
        raise NotImplementedError

    @overload  # type: ignore[override]
    def __call__(
        self,
        inputs: InputSample,
        predictions,
        **kwargs: torch.Tensor,
    ):  # noqa: D102
        ...

    @overload
    def __call__(
        self,
        inputs: List[InputSample],
        predictions,
        targets,
        **kwargs: List[torch.Tensor],
    ) -> Losses:
        ...

    def __call__(
        self,
        inputs: Union[List[InputSample], InputSample],
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
        inputs: List[InputSample],
        predictions,
        targets,
        **kwargs: List[torch.Tensor],
    ) -> Losses:
        """Process inputs, match detections with existing tracks."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self,
        inputs: InputSample,
        predictions,
        **kwargs: torch.Tensor,
    ):
        """Process inputs, match detections with existing tracks."""
        raise NotImplementedError
