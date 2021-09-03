"""Quasi-dense 3D Tracking model."""
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vist.struct import (
    Boxes2D,
    Boxes3D,
    Images,
    InputSample,
    LossesType,
    ModelOutput,
)

from .quasi_dense_rcnn import QDGeneralizedRCNNConfig, QDGeneralizedRCNN

from .base import BaseModel, BaseModelConfig, build_model
from .detect import BaseTwoStageDetector
from .detect.bbox_head import BaseBoundingBoxConfig, build_bbox_head
from .track.graph import TrackGraphConfig, build_track_graph
from .track.losses import LossConfig, build_loss
from .track.similarity import SimilarityLearningConfig, build_similarity_head
from .track.utils import cosine_similarity, split_key_ref_inputs


class QD3DTConfig(QDGeneralizedRCNNConfig):
    """Config for quasi-dense 3D tracking model."""

    bbox_3d_head: BaseBoundingBoxConfig


class QuasiDense3D(QDGeneralizedRCNN):
    """QD-3DT model class."""

    def __init__(self, cfg) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg = QD3DTConfig(**cfg.dict())
        self.detector = build_model(self.cfg.detection)
        assert isinstance(self.detector, BaseTwoStageDetector)
        self.bbox_3d_head = build_bbox_head(self.cfg.bbox_3d_head)
        self.similarity_head = build_similarity_head(self.cfg.similarity)
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.track_loss = build_loss(self.cfg.losses[0])
        self.track_loss_aux = build_loss(self.cfg.losses[1])

    def prepare_targets(
        self,
        key_inputs: List[InputSample],
        ref_inputs: List[List[InputSample]],
    ) -> Tuple[
        List[Boxes2D], List[Boxes3D], List[List[Boxes2D]], List[List[Boxes3D]]
    ]:
        """Prepare 2D and 3D targets from key / ref input samples."""
        key_targets, ref_targets = super().prepare_targets(
            key_inputs, ref_inputs
        )

        key_targets_3d = []
        key_cam_intrinsics = []
        for x in key_inputs:
            assert x.boxes3d is not None
            key_targets_3d.append(x.boxes3d.to(self.device))
            key_cam_intrinsics.append(x.intrinsics.to(self.device).tensor)

        return (
            key_targets,
            key_targets_3d,
            key_cam_intrinsics,
            ref_targets,
        )

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

        key_images, ref_images = self.prepare_images(key_inputs, ref_inputs)
        (
            key_targets,
            key_targets_3d,
            key_cam_intrinsics,
            ref_targets,
        ) = self.prepare_targets(key_inputs, ref_inputs)

        # feature extraction
        key_x = self.detector.extract_features(key_images)
        ref_x = [self.detector.extract_features(img) for img in ref_images]

        # proposal generation
        key_proposals, rpn_losses = self.detector.generate_proposals(
            key_images, key_x, key_targets
        )
        with torch.no_grad():
            ref_proposals = [
                self.detector.generate_proposals(img, x)[0]
                for img, x in zip(ref_images, ref_x)
            ]

        # bbox head
        _, roi_losses = self.detector.generate_detections(
            key_images,
            key_x,
            key_proposals,
            key_targets,
            compute_detections=False,
        )

        # 3d bbox head
        (
            bbox_3d_preds,
            pos_assigned_gt_inds,
            pos_proposals,
            _,
        ) = self.bbox_3d_head(
            key_x,
            key_proposals,
            key_targets,
            filter_negatives=True,
        )

        bbox3d_targets, labels = self.bbox_3d_head.get_targets(
            pos_proposals,
            pos_assigned_gt_inds,
            key_targets,
            key_targets_3d,
            key_cam_intrinsics,
        )

        loss_bbox_3d = self.bbox_3d_head.loss(
            bbox_3d_preds, bbox3d_targets, labels
        )

        det_losses = {**rpn_losses, **roi_losses, **loss_bbox_3d}

        # track head
        key_embeddings, key_track_targets = self.similarity_head(
            key_images,
            key_x,
            key_proposals,
            key_targets,
        )
        ref_track_targets, ref_embeddings = [], []
        for inp, x, proposal, target in zip(
            ref_images, ref_x, ref_proposals, ref_targets
        ):
            embeds, targets = self.similarity_head(inp, x, proposal, target)
            ref_embeddings += [embeds]
            ref_track_targets += [targets]

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

    def forward_test(
        self, batch_inputs: List[InputSample], postprocess: bool = True
    ) -> ModelOutput:
        """Forward function during inference."""
        import pdb

        assert len(batch_inputs) == 1, "Currently only BS=1 supported!"

        # init graph at begin of sequence
        frame_id = batch_inputs[0].metadata.frame_index
        if frame_id == 0:
            self.track_graph.reset()

        image = self.detector.preprocess_image(batch_inputs)
        cam_intrinsics = [
            x.intrinsics.to(self.device).tensor for x in batch_inputs
        ]

        # Detector
        feat = self.detector.extract_features(image)
        proposals, _ = self.detector.generate_proposals(image, feat)

        # 3D
        (bbox_3d_preds, _, _, roi_feats,) = self.bbox_3d_head(
            feat,
            proposals,
        )

        # 2D
        (
            cls_scores,
            bbox_2d_preds,
        ) = self.detector.generate_detections_from_roi_feats(roi_feats)

        (
            bbox_2d_preds,
            det_labels,
            bbox_3d_preds,
        ) = self.bbox_3d_head.get_det_bboxes(
            rois,
            cls_scores,
            bbox_2d_preds,
            bbox_3d_preds,
            img_shape,
            scale_factor,
            cfg=self.cfg.detector.test_cfg,
        )

        # similarity head
        embeddings, _ = self.similarity_head(image, feat, detections)
        if postprocess:
            ori_wh = (
                batch_inputs[0].metadata.size.width,  # type: ignore
                batch_inputs[0].metadata.size.height,  # type: ignore
            )
            self.postprocess(ori_wh, image.image_sizes[0], detections[0])

        # associate detections, update graph
        tracks = self.track_graph(detections[0], frame_id, embeddings[0])

        return dict(detect=detections, track=[tracks])
