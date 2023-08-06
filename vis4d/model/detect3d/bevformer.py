"""BEVFromer."""
from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from vis4d.common.ckpt import load_model_checkpoint
from vis4d.op.base import BaseModel
from vis4d.op.detect3d.bevformer import BEVFormerHead, GridMask
from vis4d.op.detect3d.common import Detect3DOut
from vis4d.op.fpp.fpn import FPN, LastLevelP6

REV_KEYS = [
    (r"^img_backbone\.", "basemodel."),
    (r"^img_neck.lateral_convs\.", "fpn.inner_blocks."),
    (r"^img_neck.fpn_convs\.", "fpn.layer_blocks."),
    (r"^fpn.layer_blocks.3\.", "fpn.extra_blocks.p6_conv."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


class BEVFormer(nn.Module):
    """BEVFomer."""

    def __init__(
        self,
        basemodel: BaseModel,
        bevformer_head: BEVFormerHead | None = None,
        fpn_start_index: int = 3,
    ) -> None:
        """Init."""
        super().__init__()
        self.basemodel = basemodel
        self.fpn = FPN(
            self.basemodel.out_channels[3:],
            256,
            extra_blocks=LastLevelP6(256, 256),
            start_index=fpn_start_index,
        )

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )

        self.pts_bbox_head = bevformer_head or BEVFormerHead()

        # Temporal information
        self.prev_frame_info = {
            "scene_name": None,
            "prev_bev": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

        load_model_checkpoint(
            self,
            "https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth",  # pylint: disable=line-too-long
            rev_keys=REV_KEYS,
        )

    def extract_feat(self, images: list[Tensor]) -> list[Tensor]:
        """Extract features of images."""
        n = len(images)  # N
        b = images[0].shape[0]  # B
        images = torch.stack(images, dim=1)  # [B, N, C, H, W]
        images = images.view(-1, *images.shape[2:])  # [B*N, C, H, W]

        # grid mask
        images = self.grid_mask(images)

        features = self.basemodel(images)
        # TODO: Refactor FPN to return only the features used starting from
        # start_index.
        features = self.fpn(features)[self.fpn.start_index :]
        
        img_feats = []
        for img_feat in features:
            _, c, h, w = img_feat.size()
            img_feats.append(img_feat.view(b, n, c, h, w))

        return img_feats

    def forward(
        self,
        images: list[Tensor],
        images_hw: list[list[tuple[int, int]]],
        can_bus: list[list[float]],
        scene_names: list[list[str]],
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

        img_feats = self.extract_feat(images=images)

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
