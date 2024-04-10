"""BEVFormer head."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from vis4d.data.const import AxisMode
from vis4d.op.box.box3d import transform_boxes3d
from vis4d.op.box.encoder.bevformer import NMSFreeDecoder
from vis4d.op.geometry.rotation import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    rotate_velocities,
)
from vis4d.op.layer.positional_encoding import LearnedPositionalEncoding
from vis4d.op.layer.transformer import get_clones, inverse_sigmoid
from vis4d.op.layer.weight_init import bias_init_with_prob

from ..common import Detect3DOut
from .transformer import PerceptionTransformer


def bbox3d2result(
    bbox_list: list[tuple[Tensor, Tensor, Tensor]], lidar2global: Tensor
) -> Detect3DOut:
    """Convert BEVFormer detection results to Detect3DOut.

    Args:
        bbox_list (list[tuple[Tensor, Tensor, Tensor]): List of bounding boxes,
            scores and labels.
        lidar2global (Tensor): Lidar to global transformation (B, 4, 4).

    Returns:
        Detect3DOut: Detection results.
    """
    boxes_3d = []
    velocities = []
    class_ids = []
    scores_3d = []
    for i, (bboxes, scores, labels) in enumerate(bbox_list):
        # move boxes from lidar to global coordinate system
        yaw = bboxes.new_zeros(bboxes.shape[0], 3)
        yaw[:, 2] = bboxes[:, 6]
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


class BEVFormerHead(nn.Module):
    """BEVFormer 3D detection head."""

    def __init__(
        self,
        num_classes: int = 10,
        embed_dims: int = 256,
        num_query: int = 900,
        transformer: PerceptionTransformer | None = None,
        num_reg_fcs: int = 2,
        num_cls_fcs: int = 2,
        point_cloud_range: Sequence[float] = (
            -51.2,
            -51.2,
            -5.0,
            51.2,
            51.2,
            3.0,
        ),
        bev_h: int = 200,
        bev_w: int = 200,
    ) -> None:
        """Initialize BEVFormerHead.

        Args:
            num_classes (int, optional): Number of classes. Defaults to 10.
            embed_dims (int, optional): Embedding dimensions. Defaults to 256.
            num_query (int, optional): Number of queries. Defaults to 900.
            transformer (PerceptionTransformer, optional): Transformer.
                Defaults to None. If None, a default transformer will be
                created.
            num_reg_fcs (int, optional): Number of fully connected layers in
                regression branch. Defaults to 2.
            num_cls_fcs (int, optional): Number of fully connected layers in
                classification branch. Defaults to 2.
            point_cloud_range (Sequence[float], optional): Point cloud range.
                Defaults to (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0).
            bev_h (int, optional): BEV height. Defaults to 200.
            bev_w (int, optional): BEV width. Defaults to 200.
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.positional_encoding = LearnedPositionalEncoding(
            num_feats=embed_dims // 2, row_num_embed=bev_h, col_num_embed=bev_w
        )

        self.cls_out_channels = num_classes

        self.transformer = transformer or PerceptionTransformer(
            embed_dims=embed_dims
        )

        self.code_size = 10
        self.num_query = num_query

        self.box_decoder = NMSFreeDecoder(
            num_classes=num_classes,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=300,
        )
        self.pc_range = list(point_cloud_range)
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.code_weights = nn.Parameter(
            torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                requires_grad=False,
            ),
            requires_grad=False,
        )

        self._init_layers()
        self._init_weights()

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        cls_branch: list[nn.Module] = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch: list[nn.Module] = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        fc_reg = nn.Sequential(*reg_branch)

        num_pred = self.transformer.decoder.num_layers

        self.cls_branches = get_clones(fc_cls, num_pred)
        self.reg_branches = get_clones(fc_reg, num_pred)

        self.bev_embedding = nn.Embedding(
            self.bev_h * self.bev_w, self.embed_dims
        )
        self.query_embedding = nn.Embedding(
            self.num_query, self.embed_dims * 2
        )

    def _init_weights(self) -> None:
        """Initialize weights."""
        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)

    def forward(
        self,
        mlvl_feats: list[Tensor],
        can_bus: Tensor,
        images_hw: tuple[int, int],
        cam_intrinsics: list[Tensor],
        cam_extrinsics: list[Tensor],
        lidar_extrinsics: Tensor,
        prev_bev: Tensor | None = None,
    ) -> tuple[Detect3DOut, Tensor]:
        """Forward function.

        Args:
            mlvl_feats (list[Tensor]): Features from the upstream network, each
                is with shape (B, N, C, H, W).
            can_bus (Tensor): CAN bus data, with shape (B, 18).
            images_hw (tuple[int, int]): Image height and width.
            cam_intrinsics (list[Tensor]): Camera intrinsics.
            cam_extrinsics (list[Tensor]): Camera extrinsics.
            lidar_extrinsics (list[Tensor]): LiDAR extrinsics.
            prev_bev (Tensor, optional): Previous BEV feature map, with shape
                (B, C, H, W). Defaults to None.

        Returns:
            tuple[Detect3DOut, Tensor]: Detection results and BEV feature map.
        """
        batch_size = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = bev_queries.new_zeros((batch_size, self.bev_h, self.bev_w))
        bev_pos = self.positional_encoding(bev_mask)

        bev_embed, hs, init_reference, inter_references = self.transformer(
            mlvl_feats,
            can_bus,
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            images_hw=images_hw,
            cam_intrinsics=cam_intrinsics,
            cam_extrinsics=cam_extrinsics,
            lidar_extrinsics=lidar_extrinsics,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches,
            prev_bev=prev_bev,
        )

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            outputs_coord = self.reg_branches[lvl](hs[lvl])

            assert reference.shape[-1] == 3
            outputs_coord[..., 0:2] += reference[..., 0:2]
            outputs_coord[..., 0:2] = outputs_coord[..., 0:2].sigmoid()
            outputs_coord[..., 4:5] += reference[..., 2:3]
            outputs_coord[..., 4:5] = outputs_coord[..., 4:5].sigmoid()
            outputs_coord[..., 0:1] = (
                outputs_coord[..., 0:1] * (self.pc_range[3] - self.pc_range[0])
                + self.pc_range[0]
            )
            outputs_coord[..., 1:2] = (
                outputs_coord[..., 1:2] * (self.pc_range[4] - self.pc_range[1])
                + self.pc_range[1]
            )
            outputs_coord[..., 4:5] = (
                outputs_coord[..., 4:5] * (self.pc_range[5] - self.pc_range[2])
                + self.pc_range[2]
            )

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        ret_list: list[tuple[Tensor, Tensor, Tensor]] = []
        for cls_scores, bbox_preds in zip(
            outputs_classes[-1], outputs_coords[-1]
        ):
            bboxes, scores, labels = self.box_decoder(cls_scores, bbox_preds)

            # mapping MMDetection3D's coordinate to our LIDAR coordinate
            bboxes[:, 6] = -(bboxes[:, 6] + np.pi / 2)

            ret_list.append((bboxes, scores, labels))

        return bbox3d2result(ret_list, lidar_extrinsics), bev_embed

    def __call__(
        self,
        mlvl_feats: list[Tensor],
        can_bus: Tensor,
        images_hw: tuple[int, int],
        cam_intrinsics: list[Tensor],
        cam_extrinsics: list[Tensor],
        lidar_extrinsics: Tensor,
        prev_bev: Tensor | None = None,
    ) -> tuple[Detect3DOut, Tensor]:
        """Type definition."""
        return self._call_impl(
            mlvl_feats,
            can_bus,
            images_hw,
            cam_intrinsics,
            cam_extrinsics,
            lidar_extrinsics,
            prev_bev,
        )
