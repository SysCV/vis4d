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
from .track.graph import build_track_graph
from .track.utils import split_key_ref_inputs


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
        self.track_graph = build_track_graph(self.cfg.track_graph)

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
        loss_bbox_3d = self.bbox_3d_head.forward_train(
            key_x,
            key_proposals,
            key_targets,
            key_targets_3d,
            key_cam_intrinsics,
        )

        det_losses = {**rpn_losses, **roi_losses, **loss_bbox_3d}

        # track head
        track_losses, _ = self.similarity_head.forward_train(
            [key_x, *ref_x],
            [key_proposals, *ref_proposals],
            [key_targets, *ref_targets],
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

        detections, _ = self.detector.generate_detections(
            image, feat, proposals
        )
        assert detections is not None

        bbox_2d_preds, bbox_3d_preds, keep = self.bbox_3d_head.forward_test(
            feat,
            detections,
            cam_intrinsics,
        )

        # similarity head
        embeddings = self.similarity_head.forward_test(feat, detections)
        embeddings = embeddings[0][keep]
        assert inputs[0].metadata.size is not None
        input_size = (
            inputs[0].metadata.size.width,
            inputs[0].metadata.size.height,
        )
        self.postprocess(input_size, image.image_sizes[0], bbox_2d_preds)

        # associate detections, update graph
        tracks, tracks_3d = self.track_graph(
            bbox_2d_preds, bbox_3d_preds, frame_id, embeddings
        )

        return dict(detect=[bbox_2d_preds], track=[tracks])
