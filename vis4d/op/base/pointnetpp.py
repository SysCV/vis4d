"""Pointnet++ implementation.

based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch 
Added typing and named tuples for convenience. 

#TODO write tests
"""
from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetSetAbstractionOut(NamedTuple):
    """Ouput of PointNet set abstraction."""

    coordinates: torch.Tensor  # [B, C, S]
    features: torch.Tensor  # [B, D', S]


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
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
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Indexes points.

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]

    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling.

    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples

    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
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
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = (
        torch.arange(N, dtype=torch.long)
        .to(device)
        .view(1, 1, N)
        .repeat([B, S, 1])
    )
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    B, _, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], dim=-1
        )  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


def sample_and_group_all(
    xyz: torch.Tensor, points: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample and groups all.

    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]

    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
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
        mlp: List[int],
        group_all: bool,
        norm_cls: str = "BatchNorm2d",
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
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        # Create norms
        norm_fn: Callable[[int], nn.Module] = (
            getattr(nn, norm_cls) if norm_cls is not None else None
        )

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if norm_cls is not None:
                self.mlp_bns.append(norm_fn(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def __call__(
        self, coordinates: torch.Tensor, features: torch.Tensor
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
        self, xyz: torch.Tensor, points: torch.Tensor
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
        mlp: List[int],
        norm_cls: str = "BatchNorm1d",
    ):
        """Creates a pointnet++ feature propagation layer.

        Args:
            in_channel: Number of input channels
            mlp: List with hidden dimensions of the MLP.
            norm_cls (Optional(str)): class for norm (nn.'norm_cls') or None
        """
        super(PointNetFeaturePropagation, self).__init__()
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
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: Optional[torch.Tensor],
        points2: torch.Tensor,
    ) -> torch.Tensor:
        """Call function.

        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]

        Return:
            new_points: upsampled points data, [B, D', N]
        """
        return self._call_impl(xyz1, xyz2, points1, points2)

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: Optional[torch.Tensor],
        points2: torch.Tensor,
    ) -> torch.Tensor:
        """Forward Implementation.

        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]

        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
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

    class_logits: torch.Tensor


class PointNet2Segmentation(nn.Module):  # TODO, probably move to module?
    """Pointnet++ Segmentation Network."""

    def __init__(self, num_classes: int, in_channels: int = 3):
        """Creates a new Pointnet++ for segmentation.

        Args:
            num_classes: Number of semantic classes
            in_channels: Number of input channels
        """
        super(PointNet2Segmentation, self).__init__()
        self.sa1 = PointNetSetAbstraction(
            1024, 0.1, 32, in_channels + 3, [32, 32, 64], False
        )
        self.sa2 = PointNetSetAbstraction(
            256, 0.2, 32, 64 + 3, [64, 64, 128], False
        )
        self.sa3 = PointNetSetAbstraction(
            64, 0.4, 32, 128 + 3, [128, 128, 256], False
        )
        self.sa4 = PointNetSetAbstraction(
            16, 0.8, 32, 256 + 3, [256, 256, 512], False
        )
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        self.in_channels = in_channels

    def __call__(self, xyz: torch.Tensor) -> PointNet2SegmentationOut:
        """Call implementation.

        Args:
            xyz: Pointcloud data shaped [N, n_feats, n_pts]

        Returns:
            PointNet2SegmentationOut, class logits for each point
        """
        return self._call_impl(xyz)

    def forward(self, xyz: torch.Tensor) -> PointNet2SegmentationOut:
        """Predicts the semantic class logits for each point.

        Args:
            xyz: Pointcloud data shaped [N, n_feats, n_pts]$

        Returns:
            PointNet2SegmentationOut, class logits for each point
        """
        assert xyz.size(1) == self.in_channels

        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_output = self.sa1(l0_xyz, l0_points)
        l1_xyz, l1_points = l1_output.coordinates, l1_output.features
        l2_output = self.sa2(l1_xyz, l1_points)
        l2_xyz, l2_points = l2_output.coordinates, l2_output.features
        l3_output = self.sa3(l2_xyz, l2_points)
        l3_xyz, l3_points = l3_output.coordinates, l3_output.features
        l4_output = self.sa4(l3_xyz, l3_points)
        l4_xyz, l4_points = l4_output.coordinates, l4_output.features

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        return PointNet2SegmentationOut(class_logits=x)
