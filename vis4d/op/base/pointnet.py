"""Operations for PointNet."""
from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class PointNetEncoderOut(NamedTuple):
    """Output of the PointNetEncoder."""

    features: torch.Tensor  # Feature Description [B, feature_dim]
    transformations: List[  # List with all transformation matrices [[B, d, d]]
        torch.Tensor
    ]


class LinearTransform(nn.Module):
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
        """Creates a new LinearTransform, which learns a transformation matrix
        from data

        Args:
            in_dimensions (int): input dimension
            upsampling_dims (List[int]): list of intermediate feature shapes
                                         for upsampling
            downsampling_dims (List[int]):list of intermediate feature shapes
                                          for downsampling. Make sure this
                                          matches with the last upsampling_dims
            norm_cls (Optional(str)): class for norm (nn.'norm_cls') or None
            activation_cls (str): class for activation (nn.'activation_cls')
        """

        super().__init__()
        assert len(upsampling_dims) != 0 and len(downsampling_dims) != 0
        assert upsampling_dims[-1] == downsampling_dims[0]

        self.upsampling_dims_ = upsampling_dims
        self.downsampling_dims_ = downsampling_dims
        self.in_dimension_ = in_dimension
        self.identity_ = torch.eye(in_dimension).reshape(1, in_dimension ** 2)

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
            nn.Linear(downsampling_dims[-1], in_dimension ** 2)
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
        """Linear Transform forward

        Args:
            features (Tensor[B, C, N]): Input features (e.g. points)

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


class PointNetEncoder(nn.Module):
    """PointNetEncoder.
    Encodes a pointcloud and additional features into one feature description

    Code taken from
    https://github.com/timothylimyl/PointNet-Pytorch/blob/master/pointnet/model.py
    and modified to allow for modular configuration.

    See pointnet publication for more information (https://arxiv.org/pdf/1612.00593.pdf)
    """

    def __init__(
        self,
        in_dimensions: int = 3,
        out_dimensions: int = 1024,
        mlp_dimensions: List[List[int]] = [[64, 64], [64, 128]],
        norm_cls: Optional[str] = "BatchNorm1d",
        activation_cls: str = "ReLU",
        **kwargs  # TODO, type
    ):
        """Creates a new PointNetEncoder.

        Args:
            in_dimensions (int): input dimension (e.g. 3 for xzy, 6 for xzyrgb)
            out_dimensions (int): output dimensions
            mlp_dimensions (List[List[int]]): (Dimensions of MLP layers)
            norm_cls (Optional(str)): class for norm (nn.'norm_cls') or None
            activation_cls (str): class for activation (nn.'activation_cls')
            kwargs (TODO): See arguments of @LinearTransformStn
        """
        super().__init__()

        self.out_dimension_ = out_dimensions

        # Extend dimensions to upscale from input dimension
        mlp_dimensions[0].insert(0, in_dimensions)
        mlp_dimensions[-1].append(out_dimensions)

        # Learnable transformation layers.
        self.trans_layers_ = nn.ModuleList(
            [
                LinearTransform(
                    in_dimension=dims[0],
                    norm_cls=norm_cls,
                    activation_cls=activation_cls,
                    **kwargs,
                )
                for dims in mlp_dimensions
            ]
        )

        # MLP layers
        self.mlp_layers_ = nn.ModuleList()

        # Create activation
        activation = getattr(nn, activation_cls)()

        # Create norms
        norm_fn: Callable[[int], nn.Module] = (
            getattr(nn, norm_cls) if norm_cls is not None else None
        )

        for mlp_idx, mlp_dims in enumerate(mlp_dimensions):
            layers = []

            for idx, (in_dim, out_dim) in enumerate(
                zip(mlp_dims[:-1], mlp_dims[1:])
            ):
                # Create MLP
                layers.append(torch.nn.Conv1d(in_dim, out_dim, 1))
                # Create BN if needed
                if norm_fn is not None:
                    layers.append(norm_fn(out_dim))

                # Only add activation if not last layer
                if (
                    mlp_idx != len(mlp_dimensions) - 1
                    and idx != len(mlp_dims) - 2
                ):
                    layers.append(activation)

            self.mlp_layers_.append(nn.Sequential(*layers))

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Type definition for call implementation."""
        return self._call_impl(features)

    def forward(self, features: torch.Tensor):
        """PointNetEncoder forward

        Args:
            features (Tensor[B, C, N]): Input features stacked in channels.
            e.g. raw point inputs: [B, 3, N] , w color : [B, 3+3, N], ...
        Returns:
            Extracted feature representation for input and all
            applied transformations.
        """

        transforms: List[torch.Tensor] = []

        for block_idx, trans_layer in enumerate(self.trans_layers_):
            # Apply transformation
            trans = trans_layer(features)
            transforms.append(trans)
            features = features.transpose(2, 1)
            features = torch.bmm(features, trans)
            features = features.transpose(2, 1)
            # Apply MLP
            features = self.mlp_layers_[block_idx](features)

        features = torch.max(features, 2, keepdim=True)[0]
        features = features.view(-1, self.out_dimension_)

        return PointNetEncoderOut(
            features=features, transformations=transforms
        )
