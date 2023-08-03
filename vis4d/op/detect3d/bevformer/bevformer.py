"""BEVFormer head."""
from __future__ import annotations

import copy
import torch
from torch import nn, Tensor

from vis4d.op.box.encoder.nms_free import NMSFreeDecoder
from vis4d.op.layer.weight_init import bias_init_with_prob
from vis4d.op.layer.transformer import inverse_sigmoid
from vis4d.op.layer.positional_encoding import LearnedPositionalEncoding

from .transformer import PerceptionTransformer


class BEVFormerHead(nn.Module):
    """Head of Detr3D."""

    def __init__(
        self,
        num_classes: int = 10,
        embed_dims: int = 256,
        num_query: int = 900,
        num_reg_fcs: int = 2,
        point_cloud_range: list[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        with_box_refine: bool = True,
        as_two_stage: bool = False,
        num_cls_fcs: int = 2,
        bev_h: int = 200,
        bev_w: int = 200,
    ) -> None:
        """Initialize BEVFormerHead.

        Args:
            with_box_refine (bool): Whether to refine the reference points
                in the decoder. Defaults to False.
            as_two_stage (bool) : Whether to generate the proposal from
                the outputs of encoder.
            transformer (obj:`ConfigDict`): ConfigDict is used for building
                the Encoder and Decoder.
            bev_h, bev_w (int): spatial shape of BEV queries.
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.positional_encoding = LearnedPositionalEncoding(
            num_feats=128, row_num_embed=200, col_num_embed=200
        )

        self.cls_out_channels = num_classes

        self.transformer = PerceptionTransformer(
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=embed_dims,
        )

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage

        self.code_size = 10
        self.num_query = num_query

        self.bbox_coder = NMSFreeDecoder(
            num_classes=num_classes,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=300,
        )
        self.pc_range = point_cloud_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.code_weights = nn.Parameter(
            torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                requires_grad=False,
            ),
            requires_grad=False,
        )

        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if self.as_two_stage
            else self.transformer.decoder.num_layers
        )

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)]
            )
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)]
            )

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims
            )
            self.query_embedding = nn.Embedding(
                self.num_query, self.embed_dims * 2
            )

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(
        self,
        mlvl_feats: list[Tensor],
        can_bus: Tensor,
        images_hw: list[list[tuple[int, int]]],
        cam_intrinsics: list[Tensor],
        cam_extrinsics: list[Tensor],
        lidar_extrinsics: list[Tensor],
        prev_bev: Tensor | None = None,
    ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        B = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = bev_queries.new_zeros((B, self.bev_h, self.bev_w))
        bev_pos = self.positional_encoding(bev_mask)

        bev_embed, hs, init_reference, inter_references = self.transformer(
            mlvl_feats,
            can_bus,
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            images_hw=images_hw,
            cam_intrinsics=cam_intrinsics,
            cam_extrinsics=cam_extrinsics,
            lidar_extrinsics=lidar_extrinsics,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches
            if self.with_box_refine
            else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None,
            prev_bev=prev_bev,
        )

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0])
                + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1])
                + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2])
                + self.pc_range[2]
            )

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        preds_dicts = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]

            # x,y,z,l,w,h,yaw,vx,vy
            new_boxes = bboxes.clone()
            new_boxes[:, 0] = -bboxes[:, 1]
            new_boxes[:, 1] = bboxes[:, 0]
            new_boxes[:, 3] = bboxes[:, 4]
            new_boxes[:, 4] = bboxes[:, 3]
            new_boxes[:, 7] = -bboxes[:, 8]
            new_boxes[:, 8] = bboxes[:, 7]

            scores = preds["scores"]
            labels = preds["labels"]

            ret_list.append([bboxes, scores, labels])

        return bev_embed, ret_list
