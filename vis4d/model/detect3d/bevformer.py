"""BEVFromer model implementation.

This file composes the operations associated with BEVFormer
`https://arxiv.org/abs/2203.17270` into the full model implementation.
"""

from __future__ import annotations

import copy
from typing import TypedDict

import torch
from torch import Tensor, nn

from vis4d.common.ckpt import load_model_checkpoint
from vis4d.op.base import BaseModel
from vis4d.op.detect3d.bevformer import BEVFormerHead, GridMask
from vis4d.op.detect3d.common import Detect3DOut
from vis4d.op.fpp.fpn import FPN, ExtraFPNBlock

REV_KEYS = [
    (r"^img_backbone\.", "basemodel."),
    (r"^img_neck.lateral_convs\.", "fpn.inner_blocks."),
    (r"^img_neck.fpn_convs\.", "fpn.layer_blocks."),
    (r"^fpn.layer_blocks.3\.", "fpn.extra_blocks.convs.0."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


class PrevFrameInfo(TypedDict):
    """Previous frame information."""

    scene_name: str
    prev_bev: Tensor | None
    prev_pos: Tensor
    prev_angle: Tensor


class BEVFormer(nn.Module):
    """BEVFormer 3D Detector."""

    def __init__(
        self,
        basemodel: BaseModel,
        fpn: FPN | None = None,
        pts_bbox_head: BEVFormerHead | None = None,
        weights: str | None = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            basemodel (BaseModel): Base model network.
            fpn (FPN, optional): Feature Pyramid Network. Defaults to None. If
                None, a default FPN will be used.
            pts_bbox_head (BEVFormerHead, optional): BEVFormer head. Defaults
                to None. If None, a default BEVFormer head will be used.
            weights (str, optional): Path to the checkpoint to load. Defaults
                to None.
        """
        super().__init__()
        self.basemodel = basemodel
        self.fpn = fpn or FPN(
            self.basemodel.out_channels[3:],
            256,
            extra_blocks=ExtraFPNBlock(
                extra_levels=1, in_channels=256, out_channels=256
            ),
            start_index=3,
        )

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )

        self.pts_bbox_head = pts_bbox_head or BEVFormerHead()

        # Temporal information
        self.prev_frame_info = PrevFrameInfo(
            scene_name="",
            prev_bev=None,
            prev_pos=torch.zeros(3),
            prev_angle=torch.zeros(1),
        )

        if weights is not None:
            load_model_checkpoint(self, weights, rev_keys=REV_KEYS)

    def extract_feat(self, images_list: list[Tensor]) -> list[Tensor]:
        """Extract features of images."""
        n = len(images_list)  # N
        b = images_list[0].shape[0]  # B
        images = torch.stack(images_list, dim=1)  # [B, N, C, H, W]
        images = images.view(-1, *images.shape[2:])  # [B*N, C, H, W]

        # grid mask
        if self.training:
            images = self.grid_mask(images)

        features = self.basemodel(images)
        features = self.fpn(features)[self.fpn.start_index :]

        img_feats = []
        for img_feat in features:
            _, c, h, w = img_feat.size()
            img_feats.append(img_feat.view(b, n, c, h, w))

        return img_feats

    def forward(
        self,
        images: list[Tensor],
        can_bus: list[list[float]],
        scene_names: list[str],
        cam_intrinsics: list[Tensor],
        cam_extrinsics: list[Tensor],
        lidar_extrinsics: list[Tensor],
    ) -> Detect3DOut:
        """Forward."""
        # Parse lidar extrinsics from LIDAR sensor data.
        lidar_extrinsics_tensor = lidar_extrinsics[0]
        can_bus_tensor = torch.tensor(
            can_bus, dtype=torch.float32, device=images[0].device
        )

        if scene_names[0] != self.prev_frame_info["scene_name"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None

        # update idx
        self.prev_frame_info["scene_name"] = scene_names[0]

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(can_bus_tensor[0][:3])
        tmp_angle = copy.deepcopy(can_bus_tensor[0][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            can_bus_tensor[0][:3] -= self.prev_frame_info["prev_pos"]
            can_bus_tensor[0][-1] -= self.prev_frame_info["prev_angle"]
        else:
            can_bus_tensor[0][:3] = 0
            can_bus_tensor[0][-1] = 0

        images_hw = (int(images[0].shape[-2]), int(images[0].shape[-1]))
        img_feats = self.extract_feat(images)

        out, bev_embed = self.pts_bbox_head(
            img_feats,
            can_bus_tensor,
            images_hw,
            cam_intrinsics,
            cam_extrinsics,
            lidar_extrinsics_tensor,
            prev_bev=self.prev_frame_info["prev_bev"],
        )

        # During inference, we save the BEV features and ego motion of each
        # timestamp.
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = bev_embed

        return out
