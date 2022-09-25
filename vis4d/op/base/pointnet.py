"""Operations for PointNet"""
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn


class LinearTransformStn(nn.Module):
    """Module that learns a linear transformation for a input pointcloud.

    Code taken from
    https://github.com/timothylimyl/PointNet-Pytorch/blob/master/pointnet/model.py
    and modified to allow for modular configuration.

    See T-Net in Pointnet publication (https://arxiv.org/pdf/1612.00593.pdf)
    for more information
    """

    def __init__(
        self,
        in_dimension: int = 3,
        upsampling_dims: List[int] = [64, 128, 1024],
        downsampling_dims: List[int] = [1024, 512, 256],
        norm_cls: Optional[str] = "BatchNorm1d",
        activation_cls: str = "ReLU",
    ) -> None:
        super().__init__()
        assert len(upsampling_dims) != 0 and len(downsampling_dims) != 0
        assert upsampling_dims[-1] == downsampling_dims[0]

        self.upsampling_dims_ = upsampling_dims
        self.downsampling_dims_ = downsampling_dims
        self.in_dimension_ = in_dimension
        self.identity_ = torch.eye(in_dimension).reshape(1, in_dimension**2)

        # Create activation
        self.activation_ = getattr(nn, activation_cls)()

        # Create norms
        norm_fn: Callable[[int], nn.Module] = (
            getattr(nn, norm_cls) if norm_cls is not None else None
        )

        if norm_fn is not None:
            self.norms_ = nn.ModuleList(
                norm_fn(feature_size)
                for feature_size in [*upsampling_dims, *downsampling_dims[1:]]
            )

        # Create upsampling layers
        self.upsampling_layers = nn.ModuleList(
            [nn.Conv1d(in_dimension, upsampling_dims[0], 1)]
        )
        for i in range(len(upsampling_dims) - 1):
            self.upsampling_layers.append(
                nn.Conv1d(upsampling_dims[i], upsampling_dims[i + 1], 1)
            )

        # Create downsampling layers
        self.downsampling_layers = nn.ModuleList(
            [
                nn.Linear(downsampling_dims[i], downsampling_dims[i + 1])
                for i in range(len(downsampling_dims) - 1)
            ]
        )
        self.downsampling_layers.append(
            nn.Linear(downsampling_dims[-1], in_dimension**2)
        )

    def __call__(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Type definition for call implementation."""
        return self._call_impl(features)

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """LinearTransformStn forward

        Args:
            features (Tensor[N, C]): Input features (e.g. points)

        Returns:
            Learned Canonical Transfomation Matrix for this input.
            See T-Net in Pointnet publication (https://arxiv.org/pdf/1612.00593.pdf)
            for further information
        """
        batchsize = features.shape[0]
        # Upsample features
        for idx, layer in enumerate(self.upsampling_layers):
            features = layer(features)
            if self.norms_ is not None:
                features = self.norms_[idx](features)
            features = self.activation_(features)
            print(features.shape)

        features = torch.max(features, 2, keepdim=True)[0]
        features = features.view(-1, self.upsampling_dims_[-1])

        # Downsample features
        for idx, layer in enumerate(self.downsampling_layers):
            features = layer(features)

            # Do not apply norm and activation for
            # final layer
            if idx != len(self.downsampling_layers) - 1:
                if self.norms_ is not None:
                    norm_idx = idx + len(self.upsampling_layers)
                    features = self.norms_[norm_idx](features)
                features = self.activation_(features)

        identity_batch = self.identity_.repeat(batchsize, 1)
        transformations = features + identity_batch

        return transformations.view(
            batchsize, self.in_dimension_, self.in_dimension_
        )
