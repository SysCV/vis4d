"""Base class for Vis4D segmentation models."""

import abc
from typing import List, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class BaseSegmentor(nn.Module):
    """Base segmentation head class."""

    def __init__(
        self,
        in_channels: List[int],
        channels: int,
        *,
        resize: Union[None, Tuple[int, int]] = None,
        align_corners: bool = False
    ) -> None:
        """Init.

        Args:
            in_channels (List[int]): Number of channels in multi-level image
                feature.
            channels (int): Number of output channels. Usually the number of
                classes.
            resize (Tuple[int, int]], optional): If set, the prediction maps
                will be resized to the specified size. Defaults to None.
            align_corners (bool, optional): Wether to align corners during
                interpolation. Defaults to False.
        """
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.resize = resize
        self.align_corners = align_corners

    def _upsample_feat(
        self, feat: torch.Tensor, resize: Union[None, Tuple[int, int]]
    ) -> torch.Tensor:
        """Resize and concat the features.

        Args:
            feats (List[torch.Tensor]): List of multi-level image features.
            resize (Union[None, Tuple[int, int]]): If set, the prediction maps
                will be resized to the specified size. Defaults to None.

        Returns:
            upsampled_feats (torch.Tensor): List of upsampled features.
        """
        if resize is None:
            return feat
        upsampled_feat = F.interpolate(
            input=feat,
            size=resize,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return upsampled_feat

    @abc.abstractmethod
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass during training stage.

        Args:
            x (List[torch.Tensor]): Multi-level features.

        Returns:
            predictions (List[torch.Tensor]): Pixel-level segmentation
                predictions.
        """
        raise NotImplementedError

    def __call__(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        return super().__call__(x)
