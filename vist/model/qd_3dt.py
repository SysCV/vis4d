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
from vist.model.losses import LossConfig, build_loss

from .qdtrack import QDTrackConfig, QDTrack

from .detect.bbox_head import BaseBoundingBoxConfig, build_bbox_head
from .track.utils import cosine_similarity, split_key_ref_inputs
import pdb


class QD3DTConfig(QDTrackConfig):
    """Config for quasi-dense 3D tracking model."""

    bbox_3d_head: BaseBoundingBoxConfig


class QD3DT(QDTrack):
    """QD-3DT model class."""

    def __init__(self, cfg) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg = QD3DTConfig(**cfg.dict())
        self.bbox_3d_head = build_bbox_head(self.cfg.bbox_3d_head)

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
        self,
        batch_inputs: List[List[InputSample]],
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
            filter_negatives=True,
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

    def forward_test(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Forward function during inference."""
        assert len(batch_inputs) == 1, "No reference views during test!"
        inputs = [inp[0] for inp in batch_inputs]
        assert len(inputs) == 1, "Currently only BS=1 supported!"

        # init graph at begin of sequence
        frame_id = inputs[0].metadata.frameIndex
        if frame_id == 0:
            self.track_graph.reset()

        image = self.detector.preprocess_image(inputs)

        cam_intrinsics = [x.intrinsics.to(self.device).tensor for x in inputs]

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

        # pdb.set_trace()

        (
            bbox_2d_preds,
            det_labels,
            bbox_3d_preds,
        ) = self.bbox_3d_head.get_det_bboxes(
            cls_scores, bbox_2d_preds, bbox_3d_preds
        )

        # similarity head
        embeddings, _ = self.similarity_head(image, feat, detections)
        assert inputs[0].metadata.size is not None
        input_size = (
            inputs[0].metadata.size.width,
            inputs[0].metadata.size.height,
        )
        self.postprocess(input_size, image.image_sizes[0], detections[0])

        # associate detections, update graph
        tracks = self.track_graph(detections[0], frame_id, embeddings[0])
        return dict(detect=detections, track=[tracks])
