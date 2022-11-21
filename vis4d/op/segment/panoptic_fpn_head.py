"""Panoptic FPN Head for panoptic segmentation."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConvUpsample(nn.Module):
    """ConvUpsample performs 2x upsampling after Conv.

    There are several `ConvModule` layers. In the first few layers, upsampling
    will be applied after each layer of convolution. The number of upsampling
    must be no more than the number of ConvModule layers.
    """

    def __init__(
        self,
        in_channels: int,
        inner_channels: int,
        num_layers: int = 1,
        num_upsample: None | int = None,
    ) -> None:
        """Init.

        Args:
            in_channels (int): Number of channels in the input feature map.
            inner_channels (int): Number of channels produced by the
                convolution.
            num_layers (int): Number of convolution layers.
            num_upsample (int | optional): Number of upsampling layer. Must be
                no more than num_layers. Upsampling will be applied after the
                first ``num_upsample`` layers of convolution. Default:
                ``num_layers``.
        """
        super().__init__()
        if num_upsample is None:
            num_upsample = num_layers
        assert num_upsample <= num_layers, (
            f"num_upsample ({num_upsample}) must be no more than "
            f"num_layers ({num_layers})"
        )
        self.num_layers = num_layers
        self.num_upsample = num_upsample
        self.conv = nn.ModuleList()
        for _ in range(num_layers):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        inner_channels,
                        3,
                        padding=1,
                        stride=1,
                        bias=False,
                    ),
                    nn.GroupNorm(32, inner_channels),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = inner_channels

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights."""
        for module in self.conv.modules():
            if isinstance(module, (nn.Sequential, nn.ModuleList)):
                continue
            if isinstance(module, nn.GroupNorm):
                continue
            if hasattr(module, "weight"):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if hasattr(module, "bias") and module.bias:
                    nn.init.constant_(module.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward."""
        num_upsample = self.num_upsample
        for i in range(self.num_layers):
            feats = self.conv[i](features)
            if num_upsample > 0:
                num_upsample -= 1
                feats = F.interpolate(
                    feats, scale_factor=2, mode="bilinear", align_corners=False
                )
        return feats


# TODO (thomaseh): move to op/segment
class PanopticFPNHead(nn.Module):
    """PanopticFPNHead used in Panoptic FPN.

    In this head, the number of output channels is ``num_stuff_classes
    + 1``, including all stuff classes and one thing class. The stuff
    classes will be reset from ``0`` to ``num_stuff_classes - 1``, the
    thing classes will be merged to ``num_stuff_classes``-th channel.
    """

    def __init__(
        self,
        num_classes: int = 53,
        in_channels: int = 256,
        inner_channels: int = 128,
        start_level: int = 2,
        end_level: int = 6,
    ):
        """Init.

        Args:
            num_classes (int): Number of stuff classes. Default: 53.
            in_channels (int): Number of channels in the input feature map.
            inner_channels (int): Number of channels in inner features.
            start_level (int): The start level of the input features used in
                PanopticFPN.
            end_level (int): The end level of the used features, the
                ``end_level``-th layer will not be used.
        """
        super().__init__()
        self.num_classes = num_classes

        # Used feature layers are [start_level, end_level)
        self.start_level = start_level
        self.end_level = end_level
        self.num_stages = end_level - start_level
        self.inner_channels = inner_channels

        self.conv_upsample_layers = nn.ModuleList()
        for i in range(start_level, end_level):
            self.conv_upsample_layers.append(
                ConvUpsample(
                    in_channels,
                    inner_channels,
                    num_layers=i if i > 0 else 1,
                    num_upsample=i if i > 0 else 0,
                )
            )
        self.conv_logits = nn.Conv2d(inner_channels, num_classes + 1, 1)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        nn.init.kaiming_normal_(
            self.conv_logits.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.conv_logits.bias, 0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward.

        Args:
            features (list[torch.Tensor]): Feature pyramid.

        Returns:
            torch.Tensor: Segmentation outputs.
        """
        assert self.num_stages <= len(
            features
        ), "Number of subnets must be not more than length of features."
        feats = [
            layer(features[self.start_level + i])
            for i, layer in enumerate(self.conv_upsample_layers)
        ]
        seg_preds = self.conv_logits(
            torch.sum(torch.stack(feats, dim=0), dim=0)
        )
        return seg_preds


class PanopticFPNLoss(nn.Module):
    """Panoptic FPN loss function."""

    def __init__(
        self, num_things_classes: int = 80, num_stuff_classes: int = 53
    ) -> None:
        """Init."""
        super().__init__()
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes

    def _map_thing_classes(self, gt_segs: Tensor) -> Tensor:
        """Merge thing classes to one class.

        In PanopticFPN, the stuff labels will be mapped to the range `[0,
        self.num_stuff_classes-1]`, and the thing labels will be mapped to the
        `self.num_stuff_classes`-th channel.
        """
        ntc, nsc = self.num_things_classes, self.num_stuff_classes
        fg_mask = gt_segs < ntc
        bg_mask = (gt_segs >= ntc) * (gt_segs < ntc + nsc)

        new_gt_segs = torch.clone(gt_segs)
        new_gt_segs = torch.where(bg_mask, gt_segs - ntc, new_gt_segs)
        new_gt_segs = torch.where(
            fg_mask, fg_mask.type(new_gt_segs.dtype) * nsc, new_gt_segs
        )
        return new_gt_segs

    def forward(
        self, seg_pred: torch.Tensor, target_segs: torch.Tensor
    ) -> torch.Tensor:
        """Calculate losses of Panoptic FPN head."""
        target_segs = self._map_thing_classes(target_segs)
        loss_seg = F.cross_entropy(
            seg_pred, target_segs.long(), ignore_index=255
        )
        return loss_seg


def postprocess_segms(
    segms: torch.Tensor,
    images_hw: list[tuple[int, int]],
    original_hw: list[tuple[int, int]],
) -> torch.Tensor:
    """Postprocess segmentations."""
    post_segms = []
    for segm, image_hw, orig_hw in zip(segms, images_hw, original_hw):
        post_segms.append(
            F.interpolate(
                segm[:, : image_hw[0], : image_hw[1]].unsqueeze(1),
                size=(orig_hw[0], orig_hw[1]),
                mode="bilinear",
            ).squeeze(1)
        )
    return torch.stack(post_segms).argmax(dim=1)
