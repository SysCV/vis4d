"""BEVFormer Encoder."""
from __future__ import annotations

import torch
from torch import nn, Tensor

from vis4d.op.geometry.transform import inverse_rigid_transform
from vis4d.op.layer.mlp import FFN


from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import SpatialCrossAttention


class BEVFormerEncoder(nn.Module):
    """Attention with both self and cross attention."""

    def __init__(
        self,
        num_layers: int = 6,
        num_points_in_pillar: int = 4,
        return_intermediate: bool = False,
        pc_range: list[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    ) -> None:
        """Init.

        Args:
            return_intermediate (bool): Whether to return intermediate outputs.
        """
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range

        self.layers = nn.ModuleList(
            [BEVFormerLayer() for _ in range(num_layers)]
        )

        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm
        self.eps = 1e-5

    @staticmethod
    def get_reference_points(
        H,
        W,
        Z=8,
        num_points_in_pillar=4,
        dim="3d",
        bs=1,
        device="cuda",
        dtype=torch.float,
    ):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == "3d":
            zs = (
                torch.linspace(
                    0.5,
                    Z - 0.5,
                    num_points_in_pillar,
                    dtype=dtype,
                    device=device,
                )
                .view(-1, 1, 1)
                .expand(num_points_in_pillar, H, W)
                / Z
            )
            xs = (
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                .view(1, 1, W)
                .expand(num_points_in_pillar, H, W)
                / W
            )
            ys = (
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
                .view(1, H, 1)
                .expand(num_points_in_pillar, H, W)
                / H
            )
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == "2d":
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def point_sampling(
        self,
        reference_points,
        pc_range,
        images_hw,
        cam_intrinsics,
        cam_extrinsics,
        lidar_extrinsics,
    ) -> tuple[Tensor, Tensor]:
        """Sample points from reference points.

        Run under fp32 and need to close tf32 in pytorch.
        """
        lidar2img_list = []
        for i in range(len(cam_intrinsics)):
            viewpad = torch.eye(4, device=cam_intrinsics[i].device)
            viewpad[:3, :3] = cam_intrinsics[i]

            lidar2img = (
                viewpad
                @ inverse_rigid_transform(cam_extrinsics[i])
                @ lidar_extrinsics
            )

            lidar2img_list.append(lidar2img)

        lidar2img = torch.stack(lidar2img_list, dim=1)  # (B, N, 4, 4)

        reference_points = reference_points.clone()
        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0])
            + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1])
            + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2])
            + pc_range[2]
        )

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query, _ = reference_points.shape
        num_cam = lidar2img.size(1)

        reference_points = (
            reference_points.view(D, B, 1, num_query, 4)
            .repeat(1, 1, num_cam, 1, 1)
            .unsqueeze(-1)
        )

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(
            D, 1, 1, num_query, 1, 1
        )

        reference_points_cam = torch.matmul(
            lidar2img, reference_points
        ).squeeze(-1)

        bev_mask = reference_points_cam[..., 2:3] > self.eps

        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * self.eps,
        )

        reference_points_cam[..., 0] /= 1600
        reference_points_cam[..., 1] /= 928

        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    def forward(
        self,
        bev_query: Tensor,
        value: Tensor,
        bev_h: int,
        bev_w: int,
        bev_pos: Tensor,
        spatial_shapes: int,
        level_start_index: int,
        prev_bev: Tensor | None,
        shift: Tensor,
        images_hw: list[list[tuple[int, int]]],
        cam_intrinsics: list[Tensor],
        cam_extrinsics: list[Tensor],
        lidar_extrinsics: list[Tensor],
    ):
        """Forward.

        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim="3d",
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )

        ref_2d = self.get_reference_points(
            bev_h,
            bev_w,
            dim="2d",
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )

        reference_points_img, bev_mask = self.point_sampling(
            ref_3d,
            self.pc_range,
            images_hw,
            cam_intrinsics,
            cam_extrinsics,
            lidar_extrinsics,
        )

        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(
                bs * 2, len_bev, -1
            )
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )

        for _, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                value,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_img=reference_points_img,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
            )

            bev_query = output

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class BEVFormerLayer(nn.Module):
    """Implements decoder layer in DETR transformer."""

    def __init__(
        self,
        feedforward_channels: int = 512,
        batch_first: bool = True,
        pre_norm: bool = False,
    ) -> None:
        """Init."""
        super().__init__()
        self.batch_first = batch_first
        self.pre_norm = pre_norm

        self_attn = TemporalSelfAttention()
        cross_attn = SpatialCrossAttention()

        self.attentions = nn.ModuleList()
        self.attentions.append(self_attn)
        self.attentions.append(cross_attn)

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = nn.ModuleList()

        # TODO: Try to use TransformerBlockMLP
        layers = []
        in_channels = self.embed_dims
        layers = [
            nn.Sequential(
                nn.Linear(in_channels, feedforward_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )
        ]
        in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, self.embed_dims))
        layers.append(nn.Dropout(0.1))
        layers = nn.Sequential(*layers)

        self.ffns.append(FFN(layers=layers))

        self.norms = nn.ModuleList()
        for _ in range(3):
            self.norms.append(nn.LayerNorm(self.embed_dims))

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        bev_pos: Tensor,
        ref_2d: Tensor,
        bev_h: int,
        bev_w: int,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        reference_points_img: Tensor,
        bev_mask: Tensor,
        prev_bev: Tensor | None = None,
    ) -> Tensor:
        """Forward function.

        self_attn -> norm -> cross_attn -> norm -> ffn -> norm

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """
        # Temporal self attention
        query = self.attentions[0](
            query,
            ref_2d,
            prev_bev,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            query_pos=bev_pos,
        )

        query = self.norms[0](query)

        # Spaital cross attention
        query = self.attentions[1](
            query,
            value,
            reference_points_img=reference_points_img,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            bev_mask=bev_mask,
        )

        query = self.norms[1](query)

        # FFN
        query = self.ffns[0](query)

        query = self.norms[2](query)

        return query
