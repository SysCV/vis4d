"""Pointnet++ implementation.

based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch
Added typing and named tuples for convenience.

#TODO write tests
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PointNetSetAbstractionOut(NamedTuple):
    """Ouput of PointNet set abstraction."""

    coordinates: Tensor  # [B, C, S]
    features: Tensor  # [B, D', S]


def square_distance(src: Tensor, dst: Tensor) -> Tensor:
    """Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]

    Output:
        dist: per-point square distance, [B, N, M]
    """
    bs, n_pts_in, _ = src.shape
    _, n_pts_out, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(bs, n_pts_in, 1)
    dist += torch.sum(dst**2, -1).view(bs, 1, n_pts_out)
    return dist


def index_points(points: Tensor, idx: Tensor) -> Tensor:
    """Indexes points.

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]

    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    bs = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(bs, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz: Tensor, npoint: int) -> Tensor:
    """Farthest point sampling.

    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples

    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    bs, n_pts, _ = xyz.shape
    centroids = torch.zeros(bs, npoint, dtype=torch.long).to(device)
    distance = torch.ones(bs, n_pts).to(device) * 1e10
    farthest = torch.randint(0, n_pts, (bs,), dtype=torch.long).to(device)
    batch_indices = torch.arange(bs, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(bs, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(
    radius: float, nsample: int, xyz: Tensor, new_xyz: Tensor
) -> Tensor:
    """Query around a ball with given radius.

    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]

    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    bs, n_pts_in, _ = xyz.shape
    _, n_pts_out, _ = new_xyz.shape
    group_idx = (
        torch.arange(n_pts_in, dtype=torch.long)
        .to(device)
        .view(1, 1, n_pts_in)
        .repeat([bs, n_pts_out, 1])
    )
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = n_pts_in
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = (
        group_idx[:, :, 0].view(bs, n_pts_out, 1).repeat([1, 1, nsample])
    )
    mask = group_idx == n_pts_in
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: Tensor,
    points: Tensor,
) -> tuple[Tensor, Tensor]:
    """Samples and groups.

    Input:
        npoint: Number of center to sample
        radius: Grouping Radius
        nsample: Max number of points to sample for each circle
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]

    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    bs, _, channels = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(bs, npoint, 1, channels)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], dim=-1
        )  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


def sample_and_group_all(xyz: Tensor, points: Tensor) -> tuple[Tensor, Tensor]:
    """Sample and groups all.

    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]

    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    bs, n_pts, channels = xyz.shape
    new_xyz = torch.zeros(bs, 1, channels).to(device)
    grouped_xyz = xyz.view(bs, 1, n_pts, channels)
    if points is not None:
        new_points = torch.cat(
            [grouped_xyz, points.view(bs, 1, n_pts, -1)], dim=-1
        )
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """PointNet set abstraction layer."""

    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: list[int],
        group_all: bool,
        norm_cls: str | None = "BatchNorm2d",
    ):
        """Set Abstraction Layer from the Pointnet Architecture.

        Args:
            npoint: How many points to sample
            radius: Size of the ball query
            nsample: Max number of points to group inside circle
            in_channel: Input channel dimension
            mlp: Input channel dimension of the mlp layers.
                 E.g. [32 , 32, 64] will use a MLP with three layers
            group_all: If true, groups all point inside the ball, otherwise
                       samples 'nsample' points.
            norm_cls (Optional(str)): class for norm (nn.'norm_cls') or None
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        # Create norms
        norm_fn: Callable[[int], nn.Module] | None = (
            getattr(nn, norm_cls) if norm_cls is not None else None
        )

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if norm_fn is not None:
                self.mlp_bns.append(norm_fn(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def __call__(
        self, coordinates: Tensor, features: Tensor
    ) -> PointNetSetAbstractionOut:
        """Call function.

        Input:
            coordinates: input points position data, [B, C, N]
            features: input points data, [B, D, N]

        Return:
            PointNetSetAbstractionOut with:
            coordinates: sampled points position data, [B, C, S]
            features: sample points feature data, [B, D', S]
        """
        return self._call_impl(coordinates, features)

    def forward(
        self, xyz: Tensor, points: Tensor
    ) -> PointNetSetAbstractionOut:
        """Pointnet++ set abstraction layer forward.

        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]

        Return:
            PointNetSetAbstractionOut with:
            coordinates: sampled points position data, [B, C, S]
            features: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i] if len(self.mlp_bns) != 0 else lambda x: x
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return PointNetSetAbstractionOut(new_xyz, new_points)


class PointNetFeaturePropagation(nn.Module):
    """Pointnet++ Feature Propagation Layer."""

    def __init__(
        self,
        in_channel: int,
        mlp: list[int],
        norm_cls: str = "BatchNorm1d",
    ):
        """Creates a pointnet++ feature propagation layer.

        Args:
            in_channel: Number of input channels
            mlp: list with hidden dimensions of the MLP.
            norm_cls (Optional(str)): class for norm (nn.'norm_cls') or None
        """
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        # Create norms
        norm_fn: Callable[[int], nn.Module] = (
            getattr(nn, norm_cls) if norm_cls is not None else None
        )
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            if norm_cls is not None:
                self.mlp_bns.append(norm_fn(out_channel))
            last_channel = out_channel

    def __call__(
        self,
        xyz1: Tensor,
        xyz2: Tensor,
        points1: Tensor | None,
        points2: Tensor,
    ) -> Tensor:
        """Call function.

        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points features, [B, D, N]
            points2: sampled points features, [B, D, S]

        Return:
            new_points: upsampled points data, [B, D', N]
        """
        return self._call_impl(xyz1, xyz2, points1, points2)

    def forward(
        self,
        xyz1: Tensor,
        xyz2: Tensor,
        points1: Tensor | None,
        points2: Tensor,
    ) -> Tensor:
        """Forward Implementation.

        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points features, [B, D, N]
            points2: sampled points features, [B, D, S]

        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        bs, n_pts, _ = xyz1.shape
        _, n_out_pts, _ = xyz2.shape

        if n_out_pts == 1:
            interpolated_points = points2.repeat(1, n_pts, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip: Tensor = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(bs, n_pts, 3, 1),
                dim=2,
            )

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i] if len(self.mlp_bns) != 0 else lambda x: x
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNet2SegmentationOut(NamedTuple):
    """Prediction for the pointnet++ semantic segmentation network."""

    class_logits: Tensor


class PointNet2Segmentation(nn.Module):  # TODO, probably move to module?
    """Pointnet++ Segmentation Network."""

    def __init__(self, num_classes: int, in_channels: int = 3):
        """Creates a new Pointnet++ for segmentation.

        Args:
            num_classes: Number of semantic classes
            in_channels: Number of input channels
        """
        super().__init__()

        self.set_abstractions = [
            PointNetSetAbstraction(
                1024, 0.1, 32, in_channels + 3, [32, 32, 64], False
            ),
            PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False),
            PointNetSetAbstraction(
                64, 0.4, 32, 128 + 3, [128, 128, 256], False
            ),
            PointNetSetAbstraction(
                16, 0.8, 32, 256 + 3, [256, 256, 512], False
            ),
        ]

        self.feature_propagations = [
            PointNetFeaturePropagation(768, [256, 256]),
            PointNetFeaturePropagation(384, [256, 256]),
            PointNetFeaturePropagation(320, [256, 128]),
            PointNetFeaturePropagation(128 + 3, [128, 128, 128]),
        ]

        # Final convolutions
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        self.in_channels = in_channels

    def __call__(self, xyz: Tensor) -> PointNet2SegmentationOut:
        """Call implementation.

        Args:
            xyz: Pointcloud data shaped [N, n_feats, n_pts]

        Returns:
            PointNet2SegmentationOut, class logits for each point
        """
        return self._call_impl(xyz)

    def forward(self, xyz: Tensor) -> PointNet2SegmentationOut:
        """Predicts the semantic class logits for each point.

        Args:
            xyz: Pointcloud data shaped [N, n_feats, n_pts]$

        Returns:
            PointNet2SegmentationOut, class logits for each point
        """
        assert xyz.size(1) == self.in_channels

        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        set_abstraction_out = PointNetSetAbstractionOut(
            coordinates=l0_xyz, features=l0_points
        )
        outputs: list[PointNetSetAbstractionOut] = [set_abstraction_out]

        for set_abs_layer in self.set_abstractions:
            set_abstraction_out = set_abs_layer(
                set_abstraction_out.coordinates, set_abstraction_out.features
            )

            outputs.append(set_abstraction_out)

        pointwise_features = outputs[-1].features
        for idx, feature_prop_layer in enumerate(self.feature_propagations):
            layer_after_out = outputs[-idx - 1]  # l4
            layer_out = outputs[-idx - 2]  # l3

            out_features = (
                layer_out.features if idx < len(outputs) - 1 else None
            )
            pointwise_features = feature_prop_layer(
                layer_out.coordinates,
                layer_after_out.coordinates,
                out_features,
                pointwise_features,
            )

        x = self.drop1(F.relu(self.bn1(self.conv1(pointwise_features))))
        x = self.conv2(x)
        return PointNet2SegmentationOut(class_logits=x)
