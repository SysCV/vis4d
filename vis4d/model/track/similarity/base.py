"""Vis4D base class for similarity networks."""

import abc
from typing import Dict, List, Optional, Tuple, Union, cast

import torch
from torch import nn

from vis4d.common.bbox.samplers import SamplingResult
from vis4d.struct import Boxes2D, FeatureMaps, InputSample, Losses


class BaseSimilarityHead(nn.Module):
    """Base similarity learning head class."""

    def forward(
        self,
        inputs: Union[List[InputSample], InputSample],
        boxes: Union[List[List[Boxes2D]], List[Boxes2D]],
        features: Union[
            Optional[List[Dict[str, torch.Tensor]]],
            Optional[Dict[str, torch.Tensor]],
        ] = None,
        targets=None,
    ) -> Union[
        Tuple[Losses, Optional[List[SamplingResult]]], List[torch.Tensor]
    ]:
        """Forward function of similarity head.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched, including
                possible reference views. The keyframe is at index 0.
            boxes: Detected boxes to apply similarity learning on.
            features: Input feature maps. Batched, including possible
                reference views. The keyframe is at index 0.
            targets: Targets corresponding to InputSamples.

        Returns:
            Losses: A dict of scalar loss tensors.
            Optional[List[SamplingResult]]: Sampling results. Key first, then
                reference views.
        """
        if targets is None:
            inputs = cast(InputSample, inputs)
            boxes = cast(List[Boxes2D], boxes)
            features = cast(Optional[Dict[str, torch.Tensor]], features)
            return self.forward_test(inputs, boxes, features)
        inputs = cast(List[InputSample], inputs)
        boxes = cast(List[List[Boxes2D]], boxes)
        features = cast(Optional[List[Dict[str, torch.Tensor]]], features)
        return self.forward_train(inputs, boxes, features, targets)

    @abc.abstractmethod
    def forward_train(
        self,
        inputs: List[InputSample],
        boxes: List[List[Boxes2D]],
        features: Optional[List[FeatureMaps]],
        targets,
    ) -> Tuple[Losses, Optional[List[SamplingResult]]]:
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched, including
                possible reference views. The keyframe is at index 0.
            boxes: Detected boxes to apply similarity learning on.
            features: Input feature maps. Batched, including possible
                reference views. The keyframe is at index 0.
            targets: Targets corresponding to InputSamples.

        Returns:
            Losses: A dict of scalar loss tensors.
            Optional[List[SamplingResult]]: Sampling results. Key first, then
                reference views.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        features: Optional[FeatureMaps],
    ) -> List[torch.Tensor]:
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            boxes: Input boxes to compute similarity embedding for.
            features: Input feature maps. Batched.

        Returns:
            List[torch.Tensor]: Similarity embeddings (one vector per box, one
            tensor per batch element).
        """
        raise NotImplementedError
