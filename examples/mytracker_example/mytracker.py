"""Quasi-dense instance similarity learning model."""
from typing import List, Tuple, Dict, Optional
import torch

from detectron2.modeling.meta_arch.retinanet import RetinaNet
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, Instances, ImageList
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.layers import batched_nms, cat, nonzero_tuple


import openmt.data.datasets.base
from openmt.model.detect.d2_utils import (
    images_to_imagelist,
    target_to_instance,
)
from openmt.model.track.graph import TrackGraphConfig, build_track_graph
from openmt.model.track.losses import LossConfig, build_loss
from openmt.model.track.utils import (
    cosine_similarity,
    split_key_ref_inputs,
)
from openmt.struct import Boxes2D, Images, InputSample, LossesType, ModelOutput
from openmt.model import BaseModel, BaseModelConfig
from openmt.common.bbox.samplers import SamplerConfig, build_sampler
from openmt.common.bbox.matchers import MatcherConfig, build_matcher
from openmt.model.track.similarity.base import (
    BaseSimilarityHead,
    SimilarityLearningConfig,
)


def permute_to_N_HWA_K(tensor, K: int):
    """Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)."""
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def detections_to_box2d(detections: List[Instances]) -> List[Boxes2D]:
    """Convert d2 Instances representing detections to Boxes2D.

    with class_ids
    """
    result = []
    for detection in detections:
        boxes, scores, cls = (
            detection.pred_boxes.tensor,
            detection.scores,
            detection.pred_classes,
        )
        result.append(
            Boxes2D(
                torch.cat([boxes, scores.unsqueeze(-1)], -1),
                class_ids=cls,
            )
        )
    return result


class MyTrackerConfig(BaseModelConfig):
    """Config for quasi-dense tracking model."""

    track_graph: TrackGraphConfig
    losses: List[LossConfig]
    softmax_temp: float = -1.0
    embedding_dim: int = 512
    num_classes: int


class MyTracker(BaseModel):
    """Generalized R-CNN for quasi-dense instance similarity learning."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg = MyTrackerConfig(**cfg.dict())
        self.d2_cfg = get_cfg()
        base_cfg = model_zoo.get_config_file(
            "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
        )
        self.d2_cfg.merge_from_file(base_cfg)
        self.d2_cfg.MODEL.RETINANET.NUM_CLASSES = self.cfg.num_classes
        # pylint: disable=too-many-function-args,missing-kwoa
        self.retinanet = RetinaNet(self.d2_cfg)
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.track_loss = build_loss(self.cfg.losses[0])
        self.track_loss_aux = build_loss(self.cfg.losses[1])
        self.embeding_head = torch.nn.Conv2d(
            in_channels=256,
            out_channels=self.cfg.embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.match_and_sampler = MatchSampler()

    @property
    def device(self) -> torch.device:
        """Get device where input should be moved to."""
        return self.retinanet.pixel_mean.device

    def preprocess_image(self, batched_inputs: List[InputSample]) -> Images:
        """Normalize, pad and batch the input images."""
        # images = []
        # for inp in batched_inputs:
        #     t = torch.div(inp.image.tensor, 255.0)
        #     images.append(Images(t, inp.image.image_sizes))
        # images = Images.cat(images)
        images = Images.cat([inp.image for inp in batched_inputs])
        images = images.to(self.device)
        images.tensor = (
            images.tensor - self.retinanet.pixel_mean
        ) / self.retinanet.pixel_std
        return images

    def extract_features(
        self, images: ImageList
    ) -> List[torch.Tensor]:  # type:ignore
        """Detector feature extraction stage.

        Return backbone output features
        """
        # backbone
        features = self.retinanet.backbone(images.tensor)
        features = [features[f] for f in self.retinanet.head_in_features]
        return features

    def retinanet_head(self, features, targets_d2):
        anchors = self.retinanet.anchor_generator(features)
        pred_logits, pred_anchor_deltas = self.retinanet.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [
            permute_to_N_HWA_K(x, self.retinanet.num_classes)
            for x in pred_logits
        ]
        pred_anchor_deltas = [
            permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas
        ]
        gt_labels, gt_boxes = self.retinanet.label_anchors(anchors, targets_d2)
        return (
            anchors,
            pred_logits,
            gt_labels,
            pred_anchor_deltas,
            gt_boxes,
        )

    def get_embeddings(
        self, features, anchors, gt_labels, gt_boxes, targets, pred_logits
    ):
        embs = [self.embeding_head(x) for x in features]
        embs = [emb.unsqueeze(-1).repeat(1, 1, 1, 1, 9) for emb in embs]
        embs = [
            emb.reshape(emb.shape[0], emb.shape[1], -1).permute(0, 2, 1)
            for emb in embs
        ]

        anchor_numbers = [
            t.tensor.shape[0] for t in anchors
        ]  # anchor numbers per feature map
        mask = []
        track_targets = []
        for gt_l, gt_b in zip(gt_labels, gt_boxes):
            selected_mask = torch.logical_and(
                gt_l != 0, gt_l != self.cfg.num_classes
            )
            mask.append(
                torch.split(
                    selected_mask,
                    anchor_numbers,
                )
            )
            track_targets.append(gt_b[selected_mask])
        embeddings = []
        proposals_boxes = []
        proposals_scores = []
        proposals = []
        for i, mask_per_img in enumerate(mask):
            emb_per_img = []
            anchor_per_img = []
            logits_per_img = []
            for (
                emb_per_feat,
                anchors_per_feat,
                mask_per_feat,
                logits_per_feat,
            ) in zip(embs, anchors, mask_per_img, pred_logits):
                emb_per_img.append(emb_per_feat[i][mask_per_feat])
                anchor_per_img.append(anchors_per_feat[mask_per_feat])
                logits_per_img.append(logits_per_feat[i][mask_per_feat])
            embeddings.append(torch.cat(emb_per_img, dim=0))
            logits_per_img = torch.cat(logits_per_img, dim=0)
            proposals_scores_per_img, proposals_class_per_img = torch.max(
                logits_per_img.sigmoid_(), dim=1
            )

            proposals_boxes_per_img = Boxes.cat(anchor_per_img)
            proposals.append(
                Boxes2D(
                    boxes=torch.cat(
                        (
                            proposals_boxes_per_img.tensor.to(torch.float64),
                            proposals_scores_per_img.unsqueeze(1),
                        ),
                        dim=1,
                    ),
                    class_ids=proposals_class_per_img,
                )
            )

        embeddings, track_targets = self.match_and_sampler(
            embeddings, proposals, targets  # embeddings没有值
        )
        return embeddings, track_targets

    def forward_train(
        self, batch_inputs: List[List[InputSample]]
    ) -> LossesType:
        """Forward function for training."""
        # split into key / ref pairs NxM input --> key: N, ref: Nx(M-1)
        key_inputs, ref_inputs = split_key_ref_inputs(batch_inputs)

        # group by ref views by sequence: Nx(M-1) --> (M-1)xN
        ref_inputs = [
            [ref_inputs[j][i] for j in range(len(ref_inputs))]
            for i in range(len(ref_inputs[0]))
        ]

        # prepare targets
        key_targets = [input.instances.to(self.device) for input in key_inputs]
        ref_targets = [
            [input.instances.to(self.device) for input in inputs]
            for inputs in ref_inputs
        ]

        key_images = self.preprocess_image(key_inputs)
        ref_images = [self.preprocess_image(inp) for inp in ref_inputs]
        key_images_d2 = images_to_imagelist(key_images)
        ref_images_d2 = [images_to_imagelist(img) for img in ref_images]
        key_targets_d2 = target_to_instance(
            key_targets, key_images.image_sizes  # type: ignore
        )
        ref_targets_d2 = [
            target_to_instance(ref_t, ref_img.image_sizes)  # type: ignore
            for ref_t, ref_img in zip(ref_targets, ref_images)
        ]
        # key
        key_x = self.extract_features(key_images_d2)
        (
            key_anchors,
            key_pred_logits,
            key_gt_labels,
            key_pred_anchor_deltas,
            key_gt_boxes,
        ) = self.retinanet_head(key_x, key_targets_d2)
        key_embeddings, key_track_targets = self.get_embeddings(
            key_x,
            key_anchors,
            key_gt_labels,
            key_gt_boxes,
            key_targets,
            key_pred_logits,
        )

        det_losses = self.retinanet.losses(
            key_anchors,
            key_pred_logits,
            key_gt_labels,
            key_pred_anchor_deltas,
            key_gt_boxes,
        )

        # ref

        ref_x = [self.extract_features(img) for img in ref_images_d2]
        (
            ref_anchors,
            ref_pred_logits,
            ref_gt_labels,
            ref_pred_anchor_deltas,
            ref_gt_boxes,
        ) = ([], [], [], [], [])
        for x, t in zip(ref_x, ref_targets_d2):
            (
                ref_anc,
                ref_pred_lgt,
                ref_gt_l,
                ref_pred_anc_dlt,
                ref_gt_bxs,
            ) = self.retinanet_head(x, t)
            ref_anchors.append(ref_anc)
            ref_pred_logits.append(ref_pred_lgt)
            ref_gt_labels.append(ref_gt_l)
            ref_pred_anchor_deltas.append(ref_pred_anc_dlt)
            ref_gt_boxes.append(ref_gt_bxs)
        ref_embeddings, ref_track_targets = [], []
        for r_x, r_anc, r_gt_l, r_gt_b, r_t in zip(
            ref_x, ref_anchors, ref_gt_labels, ref_gt_boxes, ref_targets
        ):
            r_embed, r_track_targets = self.get_embeddings(
                r_x, r_anc, r_gt_l, r_gt_b, r_t
            )
            ref_embeddings.append(r_embed)
            ref_track_targets.append(r_track_targets)

        track_losses = self.tracking_loss(
            key_embeddings,
            key_track_targets,
            ref_embeddings,
            ref_track_targets,
        )
        return {**det_losses, **track_losses}

    def match(
        self,
        key_embeds: Tuple[torch.Tensor],
        ref_embeds: List[Tuple[torch.Tensor]],
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """Match key / ref embeddings based on cosine similarity."""
        # for each reference view
        dists, cos_dists = [], []
        for ref_embed in ref_embeds:
            # for each batch element
            dists_curr, cos_dists_curr = [], []
            for key_embed, ref_embed_ in zip(key_embeds, ref_embed):
                dist = cosine_similarity(
                    key_embed,
                    ref_embed_,
                    normalize=False,
                    temperature=self.cfg.softmax_temp,
                )
                dists_curr.append(dist)
                if self.track_loss_aux is not None:
                    cos_dist = cosine_similarity(key_embed, ref_embed_)
                    cos_dists_curr.append(cos_dist)

            dists.append(dists_curr)
            cos_dists.append(cos_dists_curr)
        return dists, cos_dists

    @staticmethod
    def get_track_targets(
        key_targets: List[Boxes2D], ref_targets: List[List[Boxes2D]]
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """Create tracking target tensors."""
        # for each reference view
        track_targets, track_weights = [], []
        for ref_target in ref_targets:
            # for each batch element
            curr_targets, curr_weights = [], []
            for key_target, ref_target_ in zip(key_targets, ref_target):
                assert (
                    key_target.track_ids is not None
                    and ref_target_.track_ids is not None
                )
                # target shape: len(key_target) x len(ref_target_)
                target = (
                    key_target.track_ids.view(-1, 1)
                    == ref_target_.track_ids.view(1, -1)
                ).int()
                weight = (target.sum(dim=1) > 0).float()
                curr_targets.append(target)
                curr_weights.append(weight)
            track_targets.append(curr_targets)
            track_weights.append(curr_weights)
        return track_targets, track_weights

    def tracking_loss(
        self,
        key_embeddings: Tuple[torch.Tensor],
        key_targets: List[Boxes2D],
        ref_embeddings: List[Tuple[torch.Tensor]],
        ref_targets: List[List[Boxes2D]],
    ) -> LossesType:
        """Calculate losses for tracking.

        Key inputs are of type List[Tensor/Boxes2D] (Lists are length N)
        Ref inputs are of type List[List[Tensor/Boxes2D]] where the lists
        are of length MxN.
        Where M is the number of reference views and N is the
        number of batch elements.
        """
        losses = dict()

        loss_track = torch.tensor(0.0).to(self.device)
        loss_track_aux = torch.tensor(0.0).to(self.device)
        dists, cos_dists = self.match(key_embeddings, ref_embeddings)
        track_targets, track_weights = self.get_track_targets(
            key_targets, ref_targets
        )
        # for each reference view
        for curr_dists, curr_cos_dists, curr_targets, curr_weights in zip(
            dists, cos_dists, track_targets, track_weights
        ):
            # for each batch element
            for _dists, _cos_dists, _targets, _weights in zip(
                curr_dists, curr_cos_dists, curr_targets, curr_weights
            ):
                if all(_dists.shape):
                    loss_track += self.track_loss(
                        _dists,
                        _targets,
                        _weights,
                        avg_factor=_weights.sum() + 1e-5,
                    )
                    if self.track_loss_aux is not None:
                        loss_track_aux += self.track_loss_aux(
                            _cos_dists, _targets
                        )

        num_pairs = len(dists) * len(dists[0])
        losses["track_loss"] = loss_track / num_pairs
        if self.track_loss_aux is not None:
            losses["track_loss_aux"] = loss_track_aux / num_pairs

        return losses

    def forward_test(
        self, batch_inputs: List[InputSample], postprocess: bool = True
    ) -> ModelOutput:
        """Forward function during inference."""
        assert len(batch_inputs) == 1, "Currently only BS=1 supported!"

        # init graph at begin of sequence
        frame_id = batch_inputs[0].metadata.frame_index
        if frame_id == 0:
            self.track_graph.reset()

        # detector
        image = self.preprocess_image(batch_inputs)
        image_d2 = images_to_imagelist(image)
        features = self.extract_features(image_d2)

        ##########################
        anchors = self.retinanet.anchor_generator(features)
        pred_logits, pred_anchor_deltas = self.retinanet.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [
            permute_to_N_HWA_K(x, self.retinanet.num_classes)
            for x in pred_logits
        ]
        pred_anchor_deltas = [
            permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas
        ]

        embeddings = [self.embeding_head(x) for x in features]
        embeddings = [
            emb.unsqueeze(-1).repeat(1, 1, 1, 1, 9) for emb in embeddings
        ]
        embeddings = [
            emb.reshape(emb.shape[0], emb.shape[1], -1).permute(0, 2, 1)
            for emb in embeddings
        ]
        results, det_embeddings = self.inference(
            anchors,
            pred_logits,
            pred_anchor_deltas,
            image_d2.image_sizes,
            embeddings,
        )

        processed_results = []
        for results_per_image, image_size in zip(
            results, image_d2.image_sizes
        ):
            height = image_size[0]
            width = image_size[1]
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append(r)

        detections = detections_to_box2d(processed_results)

        ###########
        if postprocess:
            ori_wh = (
                batch_inputs[0].metadata.size.width,  # type: ignore
                batch_inputs[0].metadata.size.height,  # type: ignore
            )
            self.postprocess(ori_wh, image.image_sizes[0], detections[0])

        # associate detections, update graph
        tracks = self.track_graph(detections[0], frame_id, det_embeddings[0])

        return dict(detect=detections, track=[tracks])  # type:ignore

    def inference(
        self,
        anchors: List[Boxes],
        pred_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
        embeddings,
    ):
        """
        Arguments:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            pred_logits, pred_anchor_deltas: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4)
            image_sizes (List[(h, w)]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results: List[Instances] = []
        det_embeddings: List[torch.Tensor] = []
        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            embeddings_per_image = [x[img_idx] for x in embeddings]
            (
                results_per_image,
                det_embeddings_per_image,
            ) = self.inference_single_image(
                anchors,
                pred_logits_per_image,
                deltas_per_image,
                image_size,
                embeddings_per_image,
            )
            results.append(results_per_image)
            det_embeddings.append(det_embeddings_per_image)
        return results, det_embeddings

    def inference_single_image(
        self,
        anchors: List[Boxes],
        box_cls: List[torch.Tensor],
        box_delta: List[torch.Tensor],
        image_size: Tuple[int, int],
        embeddings,
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).
        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.
        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []
        embeddings_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i, embeddings_i in zip(
            box_cls, box_delta, anchors, embeddings
        ):
            # (HxWxAxK,)
            predicted_prob = box_cls_i.flatten().sigmoid_()

            # Apply two filtering below to make NMS faster.
            # 1. Keep boxes with confidence score higher than threshold
            keep_idxs = predicted_prob > self.retinanet.test_score_thresh
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = nonzero_tuple(keep_idxs)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(
                self.retinanet.test_topk_candidates, topk_idxs.size(0)
            )
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            anchor_idxs = topk_idxs // self.retinanet.num_classes
            classes_idxs = topk_idxs % self.retinanet.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            embeddings_i = embeddings_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.retinanet.box2box_transform.apply_deltas(
                box_reg_i, anchors_i.tensor
            )

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)
            embeddings_all.append(embeddings_i)

        boxes_all, scores_all, class_idxs_all, embeddings_all = [
            cat(x)
            for x in [boxes_all, scores_all, class_idxs_all, embeddings_all]
        ]
        keep = batched_nms(
            boxes_all,
            scores_all,
            class_idxs_all,
            self.retinanet.test_nms_thresh,
        )
        keep = keep[: self.retinanet.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        det_embeddings_per_image = embeddings_all[keep]
        return result, det_embeddings_per_image


class MatchSamplerConfig(BaseModelConfig):
    """Quasi-dense Similarity Head config."""

    type = "MatchSampler"
    proposal_append_gt: bool = True

    proposal_sampler: SamplerConfig = SamplerConfig(
        type="CombinedSampler",
        batch_size_per_image=256,
        positive_fraction=0.5,
        pos_strategy="instance_balanced",
        neg_strategy="iou_balanced",
    )
    proposal_matcher: MatcherConfig = MatcherConfig(
        type="MaxIoUMatcher",
        thresholds=[0.3, 0.7],
        labels=[0, -1, 1],
        allow_low_quality_matches=False,
    )


class MatchSampler(torch.nn.Module):
    """Instance embedding head for quasi-dense similarity learning."""

    def __init__(
        self,
    ) -> None:
        """Init."""
        super().__init__()
        self.cfg = MatchSamplerConfig()

        self.sampler = build_sampler(self.cfg.proposal_sampler)
        self.matcher = build_matcher(self.cfg.proposal_matcher)

    @torch.no_grad()  # type: ignore
    def match_and_sample_proposals(
        self,
        proposals: List[Boxes2D],
        targets: List[Boxes2D],
        embeddings: List[torch.Tensor],
    ) -> Tuple[List[Boxes2D], List[Boxes2D]]:
        """Match proposals to targets and subsample."""
        if self.cfg.proposal_append_gt:
            proposals = [
                Boxes2D.cat([p, t]) for p, t in zip(proposals, targets)
            ]
        matching = self.matcher.match(proposals, targets)
        return self.sampler.sample(
            matching, proposals, targets, embeddings
        )  # embeddings 没有值？

    def forward(  # type: ignore # pylint: disable=arguments-differ
        self,
        embeddings,
        proposals: List[Boxes2D],
        targets: Optional[List[Boxes2D]] = None,
        filter_negatives: bool = True,
    ) -> Tuple[Tuple[torch.Tensor], Optional[List[Boxes2D]]]:
        """Forward of embedding head."""
        assert targets is not None, "targets required during training"
        proposals, targets, embeddings = self.match_and_sample_proposals(
            proposals, targets, embeddings
        )
        if filter_negatives:
            proposals = [
                p[t.class_ids != -1] for p, t in zip(proposals, targets)  # type: ignore # pylint: disable=line-too-long
            ]
            targets = [t[t.class_ids != -1] for t in targets]  # type: ignore # pylint: disable=line-too-long
        return embeddings, targets
