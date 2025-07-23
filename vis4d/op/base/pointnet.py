"""Operations for PointNet.

Code taken from
https://github.com/timothylimyl/PointNet-Pytorch/blob/master/pointnet/model.py
and modified to allow for modular configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import NamedTuple

import torch
from torch import nn

from vis4d.common.typing import ArgsType


class PointNetEncoderOut(NamedTuple):
    """Output of the PointNetEncoder.

    features: Global features shape [N, feature_dim]
    pointwise Features: Pointwise features shape [N, last_mlp_dim, n_pts]
    transformations: list with all transformation matrixes that were used.
                     Shape [N, d, d]
    """

    features: torch.Tensor
    pointwise_features: torch.Tensor  #
    transformations: list[  # list with all transformation matrices [[B, d, d]]
        torch.Tensor
    ]


class PointNetSemanticsLoss(NamedTuple):
    """Losses for the pointnet semantic segmentation network."""

    semantic_loss: torch.Tensor
    regularization_loss: torch.Tensor


class PointNetSemanticsOut(NamedTuple):
    """Output of the PointNet Segmentation network."""

    class_logits: torch.Tensor  # B, n_classes, n_pts
    transformations: list[  # list with all transformation matrices [[B, d, d]]
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
        upsampling_dims: Iterable[int] = (64, 128, 1024),
        downsampling_dims: Iterable[int] = (1024, 512, 256),
        norm_cls: str | None = "BatchNorm1d",
        activation_cls: str = "ReLU",
    ) -> None:
        """Creates a new LinearTransform.

        This learns a transformation matrix from data.

        Args:
            in_dimension (int): input dimension
            upsampling_dims (Iterable[int]): list of intermediate feature
                                             shapes for upsampling
            downsampling_dims (Iterable[int]): list of intermediate feature
                                               shapes for downsampling.
                                               Make sure this matches with the
                                               last upsampling_dims
            norm_cls (Optional(str)): class for norm (nn.'norm_cls') or None
            activation_cls (str): class for activation (nn.'activation_cls')
        """
        super().__init__()
        self.upsampling_dims = list(upsampling_dims)
        self.downsampling_dims = list(downsampling_dims)

        assert (
            len(self.upsampling_dims) != 0 and len(self.downsampling_dims) != 0
        )
        assert self.upsampling_dims[-1] == self.downsampling_dims[0]

        self.in_dimension_ = in_dimension
        self.identity: torch.Tensor
        self.register_buffer(
            "identity", torch.eye(in_dimension).reshape(1, in_dimension**2)
        )

        # Create activation
        self.activation_ = getattr(nn, activation_cls)()

        # Create norms
        norm_fn: Callable[[int], nn.Module] | None = (
            getattr(nn, norm_cls) if norm_cls is not None else None
        )

        if norm_fn is not None:
            self.norms_ = nn.ModuleList(
                norm_fn(feature_size)
                for feature_size in (
                    *upsampling_dims,
                    *self.downsampling_dims[1:],
                )
            )

        # Create upsampling layers
        self.upsampling_layers = nn.ModuleList(
            [nn.Conv1d(in_dimension, self.upsampling_dims[0], 1)]
        )
        for i in range(len(self.upsampling_dims) - 1):
            self.upsampling_layers.append(
                nn.Conv1d(
                    self.upsampling_dims[i], self.upsampling_dims[i + 1], 1
                )
            )

        # Create downsampling layers
        self.downsampling_layers = nn.ModuleList(
            [
                nn.Linear(
                    self.downsampling_dims[i], self.downsampling_dims[i + 1]
                )
                for i in range(len(self.downsampling_dims) - 1)
            ]
        )
        self.downsampling_layers.append(
            nn.Linear(self.downsampling_dims[-1], in_dimension**2)
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
        """Linear Transform forward.

        Args:
            features (Tensor[B, C, N]): Input features (e.g. points)

        Returns:
            Learned Canonical Transfomation Matrix for this input.
            See T-Net in Pointnet publication
            (https://arxiv.org/pdf/1612.00593.pdf)
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
        features = features.view(-1, self.upsampling_dims[-1])

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

        identity_batch = self.identity.repeat(batchsize, 1)
        transformations = features + identity_batch

        return transformations.view(
            batchsize, self.in_dimension_, self.in_dimension_
        )


class PointNetEncoder(nn.Module):
    """PointNetEncoder.

    Encodes a pointcloud and additional features into one feature description

    See pointnet publication for more information
    (https://arxiv.org/pdf/1612.00593.pdf)
    """

    def __init__(
        self,
        in_dimensions: int = 3,
        out_dimensions: int = 1024,
        mlp_dimensions: Iterable[Iterable[int]] = ((64, 64), (64, 128)),
        norm_cls: str | None = "BatchNorm1d",
        activation_cls: str = "ReLU",
        **kwargs: ArgsType,
    ):
        """Creates a new PointNetEncoder.

        Args:
            in_dimensions (int): input dimension (e.g. 3 for xzy, 6 for xzyrgb)
            out_dimensions (int): output dimensions
            mlp_dimensions (Iterable[Iterable[int]]):(Dimensions of MLP layers)
            norm_cls (Optional(str)): class for norm (nn.'norm_cls') or None
            activation_cls (str): class for activation (nn.'activation_cls')
            kwargs : See arguments of @LinearTransformStn
        """
        super().__init__()

        self.out_dimension = out_dimensions

        # Extend dimensions to upscale from input dimension
        mlp_dim_list: list[list[int]] = [list(d) for d in mlp_dimensions]
        mlp_dim_list[0].insert(0, in_dimensions)
        mlp_dim_list[-1].append(out_dimensions)
        self.mlp_dimensions = mlp_dim_list

        # Learnable transformation layers.
        self.trans_layers_ = nn.ModuleList(
            [
                LinearTransform(
                    in_dimension=dims[0],
                    norm_cls=norm_cls,
                    activation_cls=activation_cls,
                    **kwargs,
                )
                for dims in mlp_dim_list
            ]
        )

        # MLP layers
        self.mlp_layers_ = nn.ModuleList()

        # Create activation
        activation = getattr(nn, activation_cls)()

        # Create norms
        norm_fn: Callable[[int], nn.Module] | None = (
            getattr(nn, norm_cls) if norm_cls is not None else None
        )

        for mlp_idx, mlp_dims in enumerate(mlp_dim_list):
            layers: list[nn.Module] = []

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
                    mlp_idx != len(mlp_dim_list) - 1
                    and idx != len(mlp_dims) - 2
                ):
                    layers.append(activation)

            self.mlp_layers_.append(nn.Sequential(*layers))

    def __call__(self, features: torch.Tensor) -> PointNetEncoderOut:
        """Type definition for call implementation."""
        return self._call_impl(features)

    def forward(self, features: torch.Tensor) -> PointNetEncoderOut:
        """Pointnet encoder forward.

        Args:
            features (Tensor[B, C, N]): Input features stacked in channels.
            e.g. raw point inputs: [B, 3, N] , w color : [B, 3+3, N], ...

        Returns:
            Extracted feature representation for input and all
            applied transformations.
        """
        transforms: list[torch.Tensor] = []

        for block_idx, trans_layer in enumerate(self.trans_layers_):
            # Apply transformation
            trans = trans_layer(features)
            transforms.append(trans)
            features = features.transpose(2, 1)
            features = torch.bmm(features, trans)
            features = features.transpose(2, 1)

            if block_idx == len(self.trans_layers_) - 1:
                pointwise_features = features.clone()

            # Apply MLP
            features = self.mlp_layers_[block_idx](features)

        features = torch.max(features, 2, keepdim=True)[0]
        features = features.view(-1, self.out_dimension)

        return PointNetEncoderOut(
            features=features,
            transformations=transforms,
            pointwise_features=pointwise_features,  # pylint: disable=possibly-used-before-assignment, line-too-long
        )


class PointNetSegmentation(nn.Module):
    """Segmentation network using a simple pointnet as encoder."""

    def __init__(
        self,
        n_classes: int,
        in_dimensions: int = 3,
        feature_dimension: int = 1024,
        norm_cls: str = "BatchNorm1d",
        activation_cls: str = "ReLU",
    ):
        """Creates a new Point Net segementation network.

        Args:
            n_classes (int): Number of semantic classes
            in_dimensions (int): Input dimension (3 for xyz, 6 xyzrgb, ...)
            feature_dimension (int): Size of feature from the encoder
            norm_cls (Optional(str)): class for norm (nn.'norm_cls') or None
            activation_cls (str): class for activation (nn.'activation_cls')

        Raises:
            ValueError: If dimensions are invalid
        """
        super().__init__()
        self.in_dimensions = in_dimensions

        self.encoder = PointNetEncoder(
            in_dimensions=in_dimensions,
            out_dimensions=feature_dimension,
            norm_cls=norm_cls,
            activation_cls=activation_cls,
        )
        pc_feat_dim = self.encoder.mlp_dimensions[-1][0]

        # Create activation
        activation = getattr(nn, activation_cls)()

        # Create norms
        norm_fn: Callable[[int], nn.Module] = (
            getattr(nn, norm_cls) if norm_cls is not None else None
        )
        self.classifier_dims = [feature_dimension + pc_feat_dim, 512, 256, 128]
        # Build Model
        self.classifier = nn.Sequential()
        for in_dim, out_dim in zip(
            self.classifier_dims[:-1], self.classifier_dims[1:]
        ):
            self.classifier.append(nn.Conv1d(in_dim, out_dim, 1))
            if norm_fn is not None:
                self.classifier.append(norm_fn(out_dim))
            self.classifier.append(activation)

        self.classifier.append(
            nn.Conv1d(
                out_dim,  # pylint: disable=undefined-loop-variable
                n_classes,
                1,
            )
        )

    def __call__(self, points: torch.Tensor) -> PointNetSemanticsOut:
        """Call function."""
        return self._call_impl(points)

    def forward(self, points: torch.Tensor) -> PointNetSemanticsOut:
        """Pointnet Segmenter Forward.

        Args:
            points (tensor) : inputs points dimension [B, in_dim, n_pts]

        Returns:
            Returns a list of tensors where the first element is
            the desired segmentation [B, n_classes, n_pts] and the other
            elements are the linear transformation matrices which
            have been used to transform the pointclouds
            @see LinearTransform
        """
        assert points.size(-2) == self.in_dimensions
        n_pts = points.size(-1)
        bs = points.size(0)
        encoder_out = self.encoder(points)
        global_features = encoder_out.features.view(bs, -1, 1).repeat(
            1, 1, n_pts
        )

        x = torch.cat([global_features, encoder_out.pointwise_features], 1)

        x = self.classifier(x)
        return PointNetSemanticsOut(
            class_logits=x, transformations=encoder_out.transformations
        )
