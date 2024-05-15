"""Semantic FPN Head for segmentation."""

from __future__ import annotations

from typing import NamedTuple

import torch.nn.functional as F
from torch import Tensor, nn

from vis4d.op.layer.conv2d import Conv2d


class SemanticFPNOut(NamedTuple):
    """Output of the SemanticFPN prediction."""

    outputs: Tensor  # logits for final prediction, (N, C, H, W)


class SemanticFPNHead(nn.Module):
    """SemanticFPNHead used in Panoptic FPN."""

    def __init__(
        self,
        num_classes: int = 53,
        in_channels: int = 256,
        inner_channels: int = 128,
        start_level: int = 2,
        end_level: int = 6,
        dropout_ratio: float = 0.1,
    ):
        """Creates an instance of the class.

        Args:
            num_classes (int): Number of classes. Default: 53.
            in_channels (int): Number of channels in the input feature map.
            inner_channels (int): Number of channels in inner features.
            start_level (int): The start level of the input features used in
                SemanticFPN.
            end_level (int): The end level of the used features, the
                ``end_level``-th layer will not be used.
            dropout_ratio (float): The drop ratio of dropout layer.
                Default: 0.1.
        """
        super().__init__()
        self.num_classes = num_classes

        # Used feature layers are [start_level, end_level)
        self.start_level = start_level
        self.end_level = end_level
        self.num_stages = end_level - start_level
        self.inner_channels = inner_channels

        self.scale_heads = nn.ModuleList()
        for i in range(start_level, end_level):
            head_length = max(1, i - start_level)
            scale_head: list[nn.Module] = []
            for k in range(head_length):
                scale_head.append(
                    Conv2d(
                        in_channels if k == 0 else inner_channels,
                        inner_channels,
                        3,
                        padding=1,
                        stride=1,
                        bias=False,
                        norm=nn.BatchNorm2d(inner_channels),
                        activation=nn.ReLU(inplace=True),
                    )
                )
                if i > start_level:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
            self.scale_heads.append(nn.Sequential(*scale_head))
        self.conv_seg = nn.Conv2d(inner_channels, num_classes, 1)
        self.dropout_ratio = dropout_ratio
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights."""
        nn.init.kaiming_normal_(
            self.conv_seg.weight, mode="fan_out", nonlinearity="relu"
        )
        if hasattr(self.conv_seg, "bias") and self.conv_seg.bias is not None:
            nn.init.constant_(self.conv_seg.bias, 0)

    def forward(self, features: list[Tensor]) -> SemanticFPNOut:
        """Transforms feature maps and returns segmentation prediction.

        Args:
            features (list[Tensor]): List of multi-level image features.

        Returns:
            SemanticFPNOut: Segmentation outputs.
        """
        assert self.num_stages <= len(
            features
        ), "Number of subnets must be not more than length of features."

        output = self.scale_heads[0](features[self.start_level])
        for i in range(1, self.num_stages):
            output = output + F.interpolate(
                self.scale_heads[i](features[self.start_level + i]),
                size=output.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        if self.dropout_ratio > 0:
            output = self.dropout(output)
        seg_preds = self.conv_seg(output)
        return SemanticFPNOut(outputs=seg_preds)

    def __call__(self, feats: list[Tensor]) -> SemanticFPNOut:
        """Type definition for function call."""
        return super()._call_impl(feats)
