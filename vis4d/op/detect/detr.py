"""Deformable DETR model and criterion classes."""
import math

import torch
import torch.nn.functional as F
from torch import nn


class DeformableDETR(nn.Module):
    """This is the Deformable DETR module that performs object detection."""

    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        num_classes: int,
        num_frames: int,
        num_queries: int,
        num_feature_levels: int,
        aux_loss: bool = True,
        with_box_refine: bool = False,
    ):
        """Initializes the model.

        Args:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
        """
        super().__init__()
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            hidden_dim,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            backbone.num_channels[0], hidden_dim, kernel_size=1
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(
                self.bbox_embed[0].layers[-1].bias.data[2:], -2.0
            )
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList(
                [self.bbox_embed for _ in range(num_pred)]
            )
            self.transformer.decoder.bbox_embed = None

    def forward(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [num_frames x 3 x H x W]
           - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels
        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        poses = []
        # print('features shape:', features.shape)
        for l, feat in enumerate(features[1:]):
            # src: [nf*N, _C, Hi, Wi],
            # mask: [nf*N, Hi, Wi],
            # pos: [nf*N, C, H_p, W_p]
            src, mask = feat.decompose()
            src_proj_l = self.input_proj[l](
                src
            )  # src_proj_l: [nf*N, C, Hi, Wi]

            # src_proj_l -> [nf, N, C, Hi, Wi]
            n, c, h, w = src_proj_l.shape
            src_proj_l = src_proj_l.reshape(
                n // self.num_frames, self.num_frames, c, h, w
            ).permute(1, 0, 2, 3, 4)

            # mask -> [nf, N, Hi, Wi]
            mask = mask.reshape(
                n // self.num_frames, self.num_frames, h, w
            ).permute(1, 0, 2, 3)

            # pos -> [nf, N, Hi, Wi]
            np, cp, hp, wp = pos[l + 1].shape
            pos_l = (
                pos[l + 1]
                .reshape(np // self.num_frames, self.num_frames, cp, hp, wp)
                .permute(1, 0, 2, 3, 4)
            )
            for n_f in range(self.num_frames):
                srcs.append(src_proj_l[n_f])
                masks.append(mask[n_f])
                poses.append(pos_l[n_f])
                assert mask is not None

        if self.num_feature_levels > (len(features) - 1):
            _len_srcs = len(features) - 1
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask  # [nf*N, H, W]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)

                # src -> [nf, N, C, H, W]
                n, c, h, w = src.shape
                src = src.reshape(
                    n // self.num_frames, self.num_frames, c, h, w
                ).permute(1, 0, 2, 3, 4)
                mask = mask.reshape(
                    n // self.num_frames, self.num_frames, h, w
                ).permute(1, 0, 2, 3)
                np, cp, hp, wp = pos_l.shape
                pos_l = pos_l.reshape(
                    np // self.num_frames, self.num_frames, cp, hp, wp
                ).permute(1, 0, 2, 3, 4)

                for n_f in range(self.num_frames):
                    srcs.append(src[n_f])
                    masks.append(mask[n_f])
                    poses.append(pos_l[n_f])

        query_embeds = None
        query_embeds = self.query_embed.weight
        (
            hs,
            memory,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer(srcs, masks, poses, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_coord
            )

        return out
