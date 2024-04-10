"""BEVFormer transformer."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.transforms.functional import rotate

from vis4d.op.layer.weight_init import xavier_init

from .decoder import BEVFormerDecoder
from .encoder import BEVFormerEncoder


class PerceptionTransformer(nn.Module):
    """Perception Transformer."""

    def __init__(
        self,
        num_cams: int = 6,
        encoder: BEVFormerEncoder | None = None,
        decoder: BEVFormerDecoder | None = None,
        embed_dims: int = 256,
        num_feature_levels: int = 4,
        rotate_center: tuple[int, int] = (100, 100),
    ) -> None:
        """Init."""
        super().__init__()
        self.num_cams = num_cams
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.rotate_center = list(rotate_center)

        self.encoder = encoder or BEVFormerEncoder(embed_dims=self.embed_dims)
        self.decoder = decoder or BEVFormerDecoder(embed_dims=self.embed_dims)

        self._init_layers()
        self._init_weights()

    def _init_layers(self) -> None:
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims)
        )
        self.reference_points = nn.Linear(self.embed_dims, 3)

        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.can_bus_mlp.add_module("norm", nn.LayerNorm(self.embed_dims))

    def _init_weights(self) -> None:
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution="uniform", bias=0.0)
        xavier_init(self.can_bus_mlp, distribution="uniform", bias=0.0)

    def get_bev_features(
        self,
        mlvl_feats: list[Tensor],
        can_bus: Tensor,
        bev_queries: Tensor,
        bev_h: int,
        bev_w: int,
        images_hw: tuple[int, int],
        cam_intrinsics: list[Tensor],
        cam_extrinsics: list[Tensor],
        lidar_extrinsics: Tensor,
        grid_length: tuple[float, float],
        bev_pos: Tensor,
        prev_bev: Tensor | None = None,
    ) -> Tensor:
        """Obtain bev features."""
        batch_size = mlvl_feats[0].shape[0]
        bev_queries = bev_queries.unsqueeze(1).repeat(1, batch_size, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = can_bus[:, 0].unsqueeze(1)
        delta_y = can_bus[:, 1].unsqueeze(1)
        ego_angle = can_bus[:, -2] / np.pi * 180

        translation_length = torch.sqrt(delta_x**2 + delta_y**2)
        translation_angle = torch.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle

        shift_y = (
            translation_length
            * torch.cos(bev_angle / 180 * np.pi)
            / grid_length[0]
            / bev_h
        )
        shift_x = (
            translation_length
            * torch.sin(bev_angle / 180 * np.pi)
            / grid_length[1]
            / bev_w
        )

        # B, xy
        shift = torch.cat([shift_x, shift_y], dim=1)

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)

            # rotate prev_bev
            for i in range(batch_size):
                rotation_angle = float(can_bus[i][-1])
                tmp_prev_bev = (
                    prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                )
                tmp_prev_bev = rotate(
                    tmp_prev_bev, rotation_angle, center=self.rotate_center
                )
                tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                    bev_h * bev_w, 1, -1
                )
                prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        bev_queries = bev_queries + self.can_bus_mlp(can_bus)[None, :, :]

        feat_flatten_list = []
        spatial_shapes_list = []
        for lvl, feat in enumerate(mlvl_feats):
            spatial_shape = feat.shape[-2:]
            feat = feat.flatten(3).permute(1, 0, 3, 2)

            # Add cams_embeds and level_embeds
            feat += self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat += self.level_embeds[None, None, lvl : lvl + 1, :].to(
                feat.dtype
            )

            spatial_shapes_list.append(spatial_shape)
            feat_flatten_list.append(feat)

        feat_flatten = torch.cat(feat_flatten_list, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes_list, dtype=torch.long, device=bev_pos.device
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        # (num_cam, H*W, bs, embed_dims)
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            images_hw=images_hw,
            cam_intrinsics=cam_intrinsics,
            cam_extrinsics=cam_extrinsics,
            lidar_extrinsics=lidar_extrinsics,
        )
        return bev_embed

    def forward(
        self,
        mlvl_feats: list[Tensor],
        can_bus: Tensor,
        bev_queries: Tensor,
        object_query_embed: Tensor,
        bev_h: int,
        bev_w: int,
        images_hw: tuple[int, int],
        cam_intrinsics: list[Tensor],
        cam_extrinsics: list[Tensor],
        lidar_extrinsics: Tensor,
        grid_length: tuple[float, float],
        bev_pos: Tensor,
        reg_branches: list[nn.Module],
        prev_bev: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward function for BEVFormer transformer.

        Args:
            mlvl_feats (list(Tensor)): Input queries from different level. Each
                element has shape [bs, num_cams, embed_dims, h, w].
            can_bus (Tensor): The can bus signals, has shape [bs, 18].
            bev_queries (Tensor): (bev_h * bev_w, embed_dims).
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, embed_dims * 2].
            bev_h (int): The height of BEV feature map.
            bev_w (int): The width of BEV feature map.
            images_hw (tuple[int, int]): The height and width of images.
            cam_intrinsics (list[Tensor]): The camera intrinsics.
            cam_extrinsics (list[Tensor]): The camera extrinsics.
            lidar_extrinsics (Tensor): The lidar extrinsics.
            grid_length (tuple[float, float]): The length of grid in x and y
                direction.
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            reg_branches (list[nn.Module]): Regression heads for feature maps
                from each decoder layer.
            prev_bev (Tensor, optional): The previous BEV feature map, has
                shape [bev_h * bev_w, bs, embed_dims]. Defaults to None.

        Returns:
            bev_embed (Tensor): BEV features has shape [bev_h *bev_w, bs,
                embed_dims].
            inter_states: Outputs from decoder has shape [1, bs, num_query,
                embed_dims].
            reference_points: As the initial reference has shape [bs,
                num_queries, 4].
            inter_references: The internal value of reference points in the
                decoder, has shape [num_dec_layers, bs,num_query, embed_dims].
        """
        # bs, bev_h*bev_w, embed_dims
        bev_embed = self.get_bev_features(
            mlvl_feats,
            can_bus,
            bev_queries,
            bev_h,
            bev_w,
            images_hw=images_hw,
            cam_intrinsics=cam_intrinsics,
            cam_extrinsics=cam_extrinsics,
            lidar_extrinsics=lidar_extrinsics,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
        )

        bs = mlvl_feats[0].shape[0]
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1
        )
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            value=bev_embed,
            reference_points=reference_points,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            query_pos=query_pos,
            reg_branches=reg_branches,
        )

        return bev_embed, inter_states, reference_points, inter_references
