"""BEVFormer Encoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from vis4d.op.geometry.transform import inverse_rigid_transform
from vis4d.op.layer.transformer import FFN, get_clones

from .spatial_cross_attention import SpatialCrossAttention
from .temporal_self_attention import TemporalSelfAttention


class BEVFormerEncoder(nn.Module):
    """Attention with both self and cross attention."""

    def __init__(
        self,
        num_layers: int = 6,
        layer: BEVFormerEncoderLayer | None = None,
        embed_dims: int = 256,
        num_points_in_pillar: int = 4,
        point_cloud_range: Sequence[float] = (
            -51.2,
            -51.2,
            -5.0,
            51.2,
            51.2,
            3.0,
        ),
        return_intermediate: bool = False,
    ) -> None:
        """Init.

        Args:
            num_layers (int): Number of layers in the encoder.
            layer (BEVFormerEncoderLayer, optional): Encoder layer. Defaults to
                None. If None, a default layer will be used.
            embed_dims (int): Embedding dimension.
            num_points_in_pillar (int): Number of points in each pillar.
            point_cloud_range (Sequence[float]): Range of the point cloud.
                Defaults to (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0).
            return_intermediate (bool): Whether to return intermediate outputs.
        """
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = point_cloud_range
        self.return_intermediate = return_intermediate

        layer = layer or BEVFormerEncoderLayer(embed_dims=embed_dims)

        self.layers = get_clones(layer, num=self.num_layers)

        self.eps = 1e-5

    def get_reference_points(
        self,
        bev_h: int,
        bev_w: int,
        dim: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Get the reference points used in SCA and TSA.

        Args:
            bev_h (int): Height of the BEV feature map.
            bev_w (int): Width of the BEV feature map.
            dim (int): Dimension of the reference points.
            batch_size (int): Batch size.
            device (torch.device): The device where reference_points should be.
            dtype (torch.dtype): The dtype of reference_points.

        Returns:
            Tensor: reference points used in decoder, has shape (batch_size,
                num_keys, num_levels, dim).
        """
        assert dim in {2, 3}, f"Unknown dim {dim}."
        # Reference points in 3D space for spatial cross-attention (SCA)
        if dim == 3:
            height_z = self.pc_range[5] - self.pc_range[2]
            zs = (
                torch.linspace(
                    0.5,
                    height_z - 0.5,
                    self.num_points_in_pillar,
                    dtype=dtype,
                    device=device,
                )
                .view(-1, 1, 1)
                .expand(self.num_points_in_pillar, bev_h, bev_w)
                / height_z
            )
            xs = (
                torch.linspace(
                    0.5, bev_w - 0.5, bev_w, dtype=dtype, device=device
                )
                .view(1, 1, bev_w)
                .expand(self.num_points_in_pillar, bev_h, bev_w)
                / bev_w
            )
            ys = (
                torch.linspace(
                    0.5, bev_h - 0.5, bev_h, dtype=dtype, device=device
                )
                .view(1, bev_h, 1)
                .expand(self.num_points_in_pillar, bev_h, bev_w)
                / bev_h
            )
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(batch_size, 1, 1, 1)
            return ref_3d

        # Reference points on 2D bev plane for temporal self-attention (TSA)
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, bev_h - 0.5, bev_h, dtype=dtype, device=device
            ),
            torch.linspace(
                0.5, bev_w - 0.5, bev_w, dtype=dtype, device=device
            ),
            indexing="ij",
        )
        ref_y = ref_y.reshape(-1)[None] / bev_h
        ref_x = ref_x.reshape(-1)[None] / bev_w
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(batch_size, 1, 1).unsqueeze(2)
        return ref_2d

    def point_sampling(
        self,
        reference_points: Tensor,
        images_hw: tuple[int, int],
        cam_intrinsics: list[Tensor],
        cam_extrinsics: list[Tensor],
        lidar_extrinsics: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Sample points from reference points."""
        lidar2img_list = []
        for i, _cam_intrinsics in enumerate(cam_intrinsics):
            viewpad = torch.eye(4, device=_cam_intrinsics.device)
            viewpad[:3, :3] = _cam_intrinsics

            lidar2img = (
                viewpad
                @ inverse_rigid_transform(cam_extrinsics[i])
                @ lidar_extrinsics
            )

            lidar2img_list.append(lidar2img)

        lidar2img = torch.stack(lidar2img_list, dim=1)  # (B, N, 4, 4)

        reference_points = reference_points.clone()
        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0])
            + self.pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1])
            + self.pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2])
            + self.pc_range[2]
        )

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        reference_points = reference_points.permute(1, 0, 2, 3)
        d, b, num_query, _ = reference_points.shape
        num_cam = lidar2img.size(1)

        reference_points = (
            reference_points.view(d, b, 1, num_query, 4)
            .repeat(1, 1, num_cam, 1, 1)
            .unsqueeze(-1)
        )

        lidar2img = lidar2img.view(1, b, num_cam, 1, 4, 4).repeat(
            d, 1, 1, num_query, 1, 1
        )

        reference_points_cam = torch.matmul(
            lidar2img, reference_points
        ).squeeze(-1)

        bev_mask = reference_points_cam[..., 2:3] > self.eps

        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.mul(
                torch.ones_like(reference_points_cam[..., 2:3]), self.eps
            ),
        )

        reference_points_cam[..., 0] /= images_hw[1]
        reference_points_cam[..., 1] /= images_hw[0]

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
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        prev_bev: Tensor | None,
        shift: Tensor,
        images_hw: tuple[int, int],
        cam_intrinsics: list[Tensor],
        cam_extrinsics: list[Tensor],
        lidar_extrinsics: Tensor,
    ) -> Tensor:
        """Forward.

        Args:
            bev_query (Tensor): Input BEV query with shape (num_query,
                batch_size, embed_dims).
            value (Tensor): Input multi-cameta features with shape (num_cam,
                num_value, batch_size, embed_dims).
            bev_h (int): BEV height.
            bev_w (int): BEV width.
            bev_pos (Tensor): BEV positional encoding with shape (batch_size,
                embed_dims).
            spatial_shapes (Tensor): Spatial shapes of multi-level
                features with shape (num_levels, 2).
            level_start_index (Tensor): Start index of each level with shape
                (num_levels, ).
            prev_bev (Tensor | None): Previous BEV features with shape
                (batch_size, embed_dims).
            shift (Tensor): Shift of each level with shape (num_levels, 2).
            images_hw (tuple[int, int]): List of image height and width.
            cam_intrinsics (list[Tensor]): List of camera intrinsics. In shape
                (num_cam, batch_size, 3, 3)
            cam_extrinsics (list[Tensor]): List of camera extrinsics. In shape
                (num_cam, batch_size, 4, 4)
            lidar_extrinsics (Tensor): LiDAR extrinsics. In shape (batch_size,
                4, 4)

        Returns:
            Tensor: Results with shape [batch_size, num_query, embed_dims]
                when return_intermediate is False, otherwise it has shape
                [num_layers, batch_size, num_query, embed_dims].
        """
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            dim=3,
            batch_size=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )

        ref_2d = self.get_reference_points(
            bev_h,
            bev_w,
            dim=2,
            batch_size=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )

        reference_points_img, bev_mask = self.point_sampling(
            ref_3d,
            images_hw,
            cam_intrinsics,
            cam_extrinsics,
            lidar_extrinsics,
        )

        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)

        batch_size, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(
                batch_size * 2, len_bev, -1
            )
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                batch_size * 2, len_bev, num_bev_level, 2
            )
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                batch_size * 2, len_bev, num_bev_level, 2
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


class BEVFormerEncoderLayer(nn.Module):
    """BEVFormer encoder layer."""

    def __init__(
        self,
        embed_dims: int = 256,
        self_attn: TemporalSelfAttention | None = None,
        cross_attn: SpatialCrossAttention | None = None,
        feedforward_channels: int = 512,
        drop_out: float = 0.1,
    ) -> None:
        """Init."""
        super().__init__()
        self.attentions = nn.ModuleList()

        self_attn = self_attn or TemporalSelfAttention(
            embed_dims=embed_dims, num_levels=1
        )
        self.attentions.append(self_attn)

        cross_attn = cross_attn or SpatialCrossAttention(embed_dims=embed_dims)
        self.attentions.append(cross_attn)

        self.embed_dims = embed_dims

        self.ffns = nn.ModuleList()
        self.ffns.append(
            FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                dropout=drop_out,
            )
        )

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
            Tensor: forwarded results with shape [num_queries, batch_size,
                embed_dims].
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
            reference_points_img,
            value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            bev_mask=bev_mask,
        )

        query = self.norms[1](query)

        # FFN
        query = self.ffns[0](query)

        query = self.norms[2](query)

        return query
