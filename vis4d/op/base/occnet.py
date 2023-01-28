from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from vis4d.op.base.conv_occnet import OccnetDecoderPrediction
from vis4d.op.geometry.positional_embedder import PositionalEmbedder
from vis4d.op.layer.mlp import ResnetBlockFC


# Max Pooling operation
def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class OccnetEncoderOut(NamedTuple):
    """Ouput of the Convolutional Occnet Encoder.

    This contains the full latent representation of a scene.
    """

    features: torch.Tensor


class ResnetPointnet(nn.Module):
    """PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    """

    def __init__(self, latent_dim=128, input_dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = latent_dim

        self.fc_pos = nn.Linear(input_dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, latent_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p) -> OccnetEncoderOut:
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return OccnetEncoderOut(features=c)


class OccnetDecoderFC(nn.Module):
    """Decoder for ConvOccnet latent representations.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=32,
        hidden_size=32,
        n_blocks=5,
        sample_mode="bilinear",
        padding=0.1,
        n_classes=1,
        activation="ReLU",
        positional_encoder=None,
    ):
        """TODO."""
        super().__init__()

        self.positional_encoder = (
            positional_encoder
            if positional_encoder is not None
            else PositionalEmbedder(add_identity=True, number_frequencies=0)
        )

        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if self.c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(self.c_dim, hidden_size) for _ in range(n_blocks)]
            )

        self.fc_p = nn.Linear(
            dim * self.positional_encoder.embedding_size, hidden_size
        )

        self.blocks = nn.ModuleList(
            [
                ResnetBlockFC(hidden_size, activation=activation)
                for _ in range(n_blocks)
            ]
        )

        self.fc_out = nn.Linear(hidden_size, n_classes)

        self.actvn = getattr(nn, activation)()
        self.sample_mode = sample_mode
        self.padding = padding

    def __call__(
        self,
        query_points: torch.Tensor,
        c_plane: OccnetEncoderOut,
        no_grad=False,
    ) -> OccnetDecoderPrediction:
        return self._call_impl(query_points, c_plane, no_grad=no_grad)

    def forward(
        self,
        query_points: torch.Tensor,
        latent: OccnetEncoderOut,
        no_grad=False,
    ) -> OccnetDecoderPrediction:

        """TODO"""

        latent = latent.features
        query_points = self.positional_encoder(query_points)
        net = self.fc_p(query_points.float())
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](latent)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        return OccnetDecoderPrediction(logits=out)


if __name__ == "__main__":
    encoder = ResnetPointnet(latent_dim=32)
    latent = encoder(torch.rand(1, 128, 3))
    decoder = OccnetDecoderFC(n_classes=5)
    print("Got latent for ", torch.rand(1, 128, 3).shape)
    print("out ", latent.features.shape)
    print("out out ", decoder(torch.rand(1, 128, 3), latent).logits.shape)
