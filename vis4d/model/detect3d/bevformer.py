"""BEVFromer."""
from __future__ import annotations

import copy
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn

from vis4d.common.ckpt import load_model_checkpoint
from vis4d.common.typing import NDArrayF32
from vis4d.data.const import AxisMode
from vis4d.op.base import BaseModel
from vis4d.op.box.box3d import transform_boxes3d
from vis4d.op.detect3d.bevformer import BEVFormerHead
from vis4d.op.fpp.fpn import FPN, LastLevelP6
from vis4d.op.geometry.rotation import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    rotate_velocities,
)

from .grid_mask import GridMask

REV_KEYS = [
    (r"^img_backbone\.", "basemodel."),
    (r"^img_neck.lateral_convs\.", "fpn.inner_blocks."),
    (r"^img_neck.fpn_convs\.", "fpn.layer_blocks."),
    (r"^fpn.layer_blocks.3\.", "fpn.extra_blocks.p6_conv."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


class Detect3DOut(NamedTuple):
    """Output of detect 3D model."""

    boxes_3d: list[Tensor]
    velocities: list[Tensor]
    class_ids: list[Tensor]
    scores_3d: list[Tensor]


def bbox3d2result(bbox_list, lidar2global: Tensor) -> Detect3DOut:
    """Convert detection results to a list of numpy arrays."""
    boxes_3d = []
    velocities = []
    class_ids = []
    scores_3d = []
    for i, (bboxes, scores, labels) in enumerate(bbox_list):
        yaw = bboxes.new_zeros(bboxes.shape[0], 3)
        yaw[:, 2] = -(bboxes[:, 6] + np.pi / 2)
        orientation = matrix_to_quaternion(euler_angles_to_matrix(yaw))

        boxes3d_lidar = torch.cat([bboxes[:, :6], orientation], dim=1)
        boxes_3d.append(
            transform_boxes3d(
                boxes3d_lidar, lidar2global[i], AxisMode.LIDAR, AxisMode.ROS
            )
        )

        _velocities = bboxes.new_zeros(bboxes.shape[0], 3)
        _velocities[:, :2] = bboxes[:, -2:]
        velocities.append(rotate_velocities(_velocities, lidar2global[i]))

        class_ids.append(labels)
        scores_3d.append(scores)

    return Detect3DOut(boxes_3d, velocities, class_ids, scores_3d)


class BEVFormer(nn.Module):
    """BEVFomer."""

    def __init__(
        self,
        use_grid_mask: bool,
        video_test_mode: bool,
        basemodel: BaseModel,
    ) -> None:
        """Init."""
        super().__init__()
        self.use_grid_mask = use_grid_mask

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "scene_name": None,
            "prev_bev": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

        self.basemodel = basemodel
        self.fpn = FPN(
            self.basemodel.out_channels[3:],
            256,
            extra_blocks=LastLevelP6(256, 256),
            start_index=0,
        )

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )

        self.pts_bbox_head = BEVFormerHead()

        load_model_checkpoint(
            self,
            "vis4d-workspace/checkpoints/bevformer_r101_dcn_24ep.pth",
            map_location="cpu",
            rev_keys=REV_KEYS,
        )

    def extract_feat(self, images: list[Tensor]) -> list[Tensor]:
        """Extract features of images."""
        N = len(images)
        B = images[0].shape[0]
        images = torch.cat(images)  # [N * B, C, H, W]

        if self.use_grid_mask:
            images = self.grid_mask(images)

        features = self.basemodel(images)
        features = self.fpn(features)

        img_feats = []
        for img_feat in features:
            BN, C, H, W = img_feat.size()
            img_feats.append(img_feat.view(B, N, C, H, W))

        return img_feats

    def forward(
        self,
        images: list[Tensor],
        images_hw: list[list[tuple[int, int]]],
        can_bus: list[NDArrayF32],
        scene_names: list[list[str]],
        cam_intrinsics: list[Tensor],
        cam_extrinsics: list[Tensor],
        lidar_extrinsics: list[Tensor],
    ) -> Detect3DOut:
        """Forward."""
        # FIXME: Refactor ToTensor transform
        lidar_extrinsics = lidar_extrinsics[0]
        can_bus = torch.cat(
            [torch.from_numpy(c).unsqueeze(0) for c in can_bus]
        ).to(images[0].device)

        if scene_names[0] != self.prev_frame_info["scene_name"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None

        # update idx
        self.prev_frame_info["scene_name"] = scene_names[0]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(can_bus[0][:3])
        tmp_angle = copy.deepcopy(can_bus[0][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            can_bus[0][:3] -= self.prev_frame_info["prev_pos"]
            can_bus[0][-1] -= self.prev_frame_info["prev_angle"]
        else:
            can_bus[0][:3] = 0
            can_bus[0][-1] = 0

        img_feats = self.extract_feat(images=images)

        bev_embed, bbox_list = self.pts_bbox_head(
            img_feats,
            can_bus,
            images_hw,
            cam_intrinsics,
            cam_extrinsics,
            lidar_extrinsics,
            prev_bev=self.prev_frame_info["prev_bev"],
        )

        # During inference, we save the BEV features and ego motion of each
        # timestamp.
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = bev_embed

        return bbox3d2result(bbox_list, lidar_extrinsics)


def filer_distance(boxes3d: Tensor, class_ids: Tensor) -> Tensor:
    """Filter boxes3d on distance."""
    from vis4d.data.datasets.nuscenes import (
        nuscenes_class_map,
        nuscenes_detection_range_map,
    )

    inv_class_map = {v: k for k, v in nuscenes_class_map.items()}

    det_range = torch.tensor(
        [
            nuscenes_detection_range_map[inv_class_map[int(class_id)]]
            for class_id in class_ids
        ]
    ).to(boxes3d.device)

    radius = torch.linalg.norm(boxes3d[:, :2], dim=1)

    return radius <= det_range
