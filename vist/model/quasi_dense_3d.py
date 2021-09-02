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
    motion: str = "kf3d"


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
            bbox_3d_pred,
            pos_assigned_gt_inds,
            pos_proposals,
        ) = self.bbox_3d_head(
            key_x,
            key_proposals,
            key_targets,
            self.detector.mm_detector.roi_head,
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
            bbox_3d_pred, bbox3d_targets, labels
        )

        det_losses = {**rpn_losses, **roi_losses, **loss_bbox_3d}

        # track head
        # TODO: Adding qd-3dt tracking graph
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
