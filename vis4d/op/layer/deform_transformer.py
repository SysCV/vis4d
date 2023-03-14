import copy
import math
from typing import List, NamedTuple, Optional

import torch
import torch.nn.functional as F

# from models.ops.modules import MSDeformAttn
from torch import nn
from torch.nn.init import constant_, normal_, uniform_, xavier_uniform_

from vis4d.op.loss.common import inverse_sigmoid
from vis4d.op.util import clone


class DTEncoderOut(NamedTuple):
    """Deformable Transformer's encoder output."""


class DTDecoderOut(NamedTuple):
    """Deformable Transformer's encoder output."""


class DeformableTransformer(nn.Module):
    """Deformable Transformer."""

    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        return_intermediate_dec: bool = False,
        num_frames: int = 1,
        num_feature_levels: int = 4,
        dec_n_points: int = 4,
        enc_n_points: int = 4,
    ):
        """Initialize Deformable Transformer.

        Args:
            d_model (int): the number of expected features in the
                encoder/decoder inputs. Defaults to 256.
            n_head (int): the number of heads in the multi-head attention
                models. Defaults to 8.
            num_encoder_layers (int): the number of sub-encoder-layers in the
                encoder. Defaults to 6.
            num_decoder_layers (int): the number of sub-decoder-layers in the
                decoder. Defaults to 6.
            dim_feedforward (int): the dimension of the feed-forward network
                model. Defaults to 1024.
            dropout (float): the dropout value. Defaults to 0.1.
            activation (str): the activation function of intermediate layer,
                relu or gelu. Defaults to "relu".
            return_intermediate_dec (bool): if True, return the output of
                each decoder layer. Defaults to False.
            num_frames (int): number of frames. Defaults to 1.
            num_feature_levels (int): number of feature levels. Defaults to 4.
            dec_n_points (int): number of sampling points for
                deformable attention in decoder. Defaults to 4.
            enc_n_points (int): number of sampling points for deformable
                attention in encoder. Defaults to 4.
        """
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.num_feature_levels = num_feature_levels

        # Set up encoder
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            n_head,
            enc_n_points,
        )
        self.encoder = DeformableTransformerEncoder(
            encoder_layer, num_encoder_layers
        )

        # Set up decoder
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            n_head,
            dec_n_points,
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate_dec
        )

        # Set up reference points
        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, d_model)
        )
        self.reference_points = nn.Linear(d_model, 2)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert query_embed is not None
        # srcs: 4(N, C, Hi, Wi)
        # query_embed: [300, C]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(
            zip(srcs, masks, pos_embeds)
        ):
            bs, nf, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(3).transpose(2, 3)  # src: [N, nf, Hi*Wi, C]
            mask = mask.flatten(2)  # mask: [N, nf, Hi*Wi]
            pos_embed = pos_embed.flatten(3).transpose(
                2, 3
            )  # pos_embed: [N, nf, Hp*Wp, C]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, 1, -1)
            # TODO: add temporal embed for different frames' feature. Since frames is not fixed, it should be hard encoded

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        # src_flatten: [\sigma(N*Hi*Wi), C]
        src_flatten = torch.cat(src_flatten, 2)
        mask_flatten = torch.cat(mask_flatten, 2)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m[:, 0]) for m in masks], 1
        )

        # encoder
        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )
        # src_flatten,lvl_pos_embed_flatten shape= [bz, nf, 4lvl*wi*hi, C]    mask_flatten: [bz, nf, 4lvl*wi*hi]

        # prepare input for decoder
        bs, nf, _, c = memory.shape

        query_embed, target = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        target = target.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        reference_points = reference_points.unsqueeze(1).repeat(
            1, nf, 1, 1
        )  # [bz,nf,300,2]
        init_reference_out = reference_points

        # decoder
        hs, hs_box, inter_references, inter_samples = self.decoder(
            target,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_embed,
            mask_flatten,
        )

        return DTOut(
            hs=hs,
            hs_box=hs_box,
            memory=memory,
            init_reference_out=init_reference_out,
            inter_references=inter_references,
            inter_samples=inter_samples,
            valid_ratios=valid_ratios,
        )
        # return (
        #     hs,
        #     hs_box,
        #     memory,
        #     init_reference_out,
        #     inter_references,
        #     inter_samples,
        #     None,
        #     valid_ratios,
        # )

    def __call__(
        self,
        srcs: List[torch.Tensor],
        masks: List[torch.Tensor],
        pos_embeds: List[torch.Tensor],
        query_embed: torch.Tensor,
    ) -> DTDecoderOut:
        """Type definition for call implementation."""
        return self._call_impl(srcs, masks, pos_embeds, query_embed)


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(
            d_model, n_levels, n_heads, n_points, "encode"
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            None,
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    """Multi-layer Deformable Transformer encoder."""

    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        """Initilize Deformable Transformer encoder.

        Args:
            encoder_layer (nn.Module): an instance of the
                DeformableTransformerEncoderLayer.
            num_layers (int): the number of Deformable Transformer encoder
                layers.
        """
        super().__init__()
        self.layers = clone(encoder_layer, num_clones=num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(
        spatial_shapes: list[tuple[float, float]],
        valid_ratios: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Get reference points for each level.

        Args:
            spatial_shapes (list[tuple]): Spatial shape of each level.
            valid_ratios (torch.Tensor): Valid ratios of each level. Shape
                [1, num_levels, 2].
            device (torch.device): Device of reference points.

        Returns:
            torch.Tensor: Reference points of each level. Shape [1, num_levels,
                num_points, 2].
        """
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H_ - 0.5, H_, dtype=torch.float32, device=device
                ),
                torch.linspace(
                    0.5, W_ - 0.5, W_, dtype=torch.float32, device=device
                ),
            )
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H_
            )
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W_
            )
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        src: torch.Tensor,
        spatial_shapes: list[tuple[int, int]],
        level_start_index: list[int],
        valid_ratios: list[int],
        pos: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tensor:
        """Forward function for Deformable Transformer encoder.

        Args:
            src (torch.Tensor): Input feature map with the shape of [num_query,
                bs, embed_dims].
            spatial_shapes (list[tuple]): Spatial shape of features from each
                level. e.g. [(40, 36), (20, 18), (10, 9), (5, 4), (3, 3),
                (2, 2)] for FPN.
            level_start_index (list[int]): Start index of each level from the
        """
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        for _, layer in enumerate(self.layers):
            output = layer(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        """Initialize Deformable Transformer Decoder Layer.

        Args:
            d_model (int): dimension of model. Defaults to 256.
            d_ffn (int): dimension of feed forward network. Defaults to 1024.
            dropout (float): dropout rate. Defaults to 0.1.
            activation (str): activation function. Defaults to "relu".
            n_levels (int): number of feature levels. Defaults to 4.
            n_heads (int): number of heads. Defaults to 8.
            n_points (int): number of sampling points for
                each head in each level. Defaults to 4.
        """
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(
            d_model, n_levels, n_heads, n_points, "decode"
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1_box = nn.Dropout(dropout)
        self.norm1_box = nn.LayerNorm(d_model)

        # self attention for mask&class query
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # self attention for box query
        self.self_attn_box = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout2_box = nn.Dropout(dropout)
        self.norm2_box = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # ffn for box
        self.linear1_box = nn.Linear(d_model, d_ffn)
        self.activation_box = _get_activation_fn(activation)
        self.dropout3_box = nn.Dropout(dropout)
        self.linear2_box = nn.Linear(d_ffn, d_model)
        self.dropout4_box = nn.Dropout(dropout)
        self.norm3_box = nn.LayerNorm(d_model)

        self.time_attention_weights = nn.Linear(d_model, 1)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def with_pos_embed_multf(
        tensor, pos
    ):  # boardcase pos to every frame features
        return tensor if pos is None else tensor + pos.unsqueeze(1)

    def forward_ffn(self, target):
        target_2 = self.linear2(
            self.dropout3(self.activation(self.linear1(target)))
        )
        target = target + self.dropout4(target_2)
        target = self.norm3(target)
        return target

    def forward_ffn_box(self, target):
        target_2 = self.linear2_box(
            self.dropout3_box(self.activation_box(self.linear1_box(target)))
        )
        target = target + self.dropout4_box(target_2)
        target = self.norm3_box(target)
        return target

    def forward(
        self,
        target,
        tgt_box,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
    ):
        # self attention
        q1 = k1 = self.with_pos_embed(target, query_pos)
        target_2 = self.self_attn(
            q1.transpose(0, 1), k1.transpose(0, 1), target.transpose(0, 1)
        )[0].transpose(0, 1)
        target = target + self.dropout2(target_2)
        target = self.norm2(target)

        if len(tgt_box.shape) == 3:  # tgt_box [bz,300,C]. # first layer
            # box-target self attention
            q_box = k_box = self.with_pos_embed(tgt_box, query_pos)
            tgt2_box = self.self_attn_box(
                q_box.transpose(0, 1),
                k_box.transpose(0, 1),
                tgt_box.transpose(0, 1),
            )[0].transpose(0, 1)
            tgt_box = tgt_box + self.dropout2_box(tgt2_box)
            tgt_box = self.norm2_box(tgt_box)
            # cross attention
            (
                target_2,
                tgt2_box,
                sampling_locations,
                attention_weights,
            ) = self.cross_attn(
                self.with_pos_embed(target, query_pos),
                self.with_pos_embed(tgt_box, query_pos),
                reference_points,
                src,
                src_spatial_shapes,
                level_start_index,
                src_padding_mask,
            )

        else:  # tgt_box [bz,nf, 300,C]
            assert len(tgt_box.shape) == 4
            N, nf, num_q, C = tgt_box.shape
            # self attention
            tgt_list = []
            for i_f in range(nf):
                tgt_box_i = tgt_box[:, i_f]
                q_box = k_box = self.with_pos_embed(tgt_box_i, query_pos)
                tgt2_box_i = self.self_attn_box(
                    q_box.transpose(0, 1),
                    k_box.transpose(0, 1),
                    tgt_box_i.transpose(0, 1),
                )[0].transpose(0, 1)
                tgt_box_i = tgt_box_i + self.dropout2_box(tgt2_box_i)
                tgt_box_i = self.norm2_box(tgt_box_i)
                tgt_list.append(tgt_box_i.unsqueeze(1))
            tgt_box = torch.cat(tgt_list, dim=1)

            # cross attention
            (
                target_2,
                tgt2_box,
                sampling_locations,
                attention_weights,
            ) = self.cross_attn(
                self.with_pos_embed(target, query_pos),
                self.with_pos_embed_multf(tgt_box, query_pos),
                reference_points,
                src,
                src_spatial_shapes,
                level_start_index,
                src_padding_mask,
            )

        if len(tgt_box.shape) == 3:
            tgt_box = tgt_box.unsqueeze(1) + self.dropout1_box(tgt2_box)
        else:
            tgt_box = tgt_box + self.dropout1_box(tgt2_box)
        tgt_box = self.norm1_box(tgt_box)
        # ffn box
        tgt_box = self.forward_ffn_box(tgt_box)

        time_weight = self.time_attention_weights(tgt_box)
        time_weight = F.softmax(time_weight, 1)
        target_2 = (target_2 * time_weight).sum(1)

        target = target + self.dropout1(target_2)
        target = self.norm1(target)
        # ffn
        target = self.forward_ffn(target)

        return target, tgt_box, sampling_locations, attention_weights


class DeformableTransformerDecoder(nn.Module):
    """Implements the Deformable Transformer decoder."""

    def __init__(
        self,
        decoder_layer: DeformableTransformerDecoderLayer,
        num_layers: int,
        return_intermediate: bool = False,
    ) -> None:
        """Initialize Deformable Transformer decoder.

        Args:
            decoder_layer (DeformableTransformerDecoderLayer): an instance of
                the DeformableTransformerDecoderLayer() class.
            num_layers (int): the number of sub-decoder-layers in the decoder.
            return_intermediate (bool): if True, return the output of each
                decoder layer, instead of the last layer.
        """
        super().__init__()
        self.layers = clone(decoder_layer, num_clones=num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None

    def forward(
        self,
        target: torch.Tensor,
        reference_points: torch.Tensor,
        src: torch.Tensor,
        src_spatial_shapes: list[tuple[int, int]],
        src_level_start_index: list[int],
        src_valid_ratios: list[tuple[int, int]],
        query_pos: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
    ):
        output = target
        intermediate = []  # save mask and class query in each decoder layers
        intermediate_box = []  # save box query
        intermediate_reference_points = []
        intermediate_samples = []

        # reference_points： [bz, nf, 300, 2]
        # src: [2, nf, len_q, 256] encoder output

        output_box = target  # box and mask&class share the same initial target, but perform deformable attention across frames independently,
        # before first decoder layer, output_box is  [bz,300,C]
        # after the first deformable attention, output_box becomes [bz, nf, 300, C] and keep shape between each decoder layers

        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[
                        :, None, None
                    ]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, :, None]
                    * src_valid_ratios[:, None, None]
                )
                # reference_points_input [bz, nf, 300, 4, 2]

            output, output_box, sampling_locations, attention_weights = layer(
                output,
                output_box,
                query_pos,
                reference_points_input,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_id](output_box)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points
                    )
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                        ..., :2
                    ] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_box.append(output_box)
                intermediate_reference_points.append(reference_points)
                # intermediate_samples.append(samples_keep)

        # if self.return_intermediate:
        #     return (
        #         torch.stack(intermediate),
        #         torch.stack(intermediate_box),
        #         torch.stack(intermediate_reference_points),
        #         None,
        #     )
        return DTDecoderOut(
            output=output,
            reference_points=reference_points,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
        )

        # return output, reference_points
