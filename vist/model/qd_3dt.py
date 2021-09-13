"""Quasi-dense 3D Tracking model."""
from typing import List

import torch

from vist.struct import InputSample, LossesType, ModelOutput

from .detect.roi_head import BaseRoIHeadConfig, build_roi_head
from .qdtrack import QDTrack, QDTrackConfig
from .track.graph import build_track_graph
import pdb


class QD3DTConfig(QDTrackConfig):
    """Config for quasi-dense 3D tracking model."""

    bbox_3d_head: BaseRoIHeadConfig


class QD3DT(QDTrack):
    """QD-3DT model class."""

    def __init__(self, cfg) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg = QD3DTConfig(**cfg.dict())
        self.bbox_3d_head = build_roi_head(self.cfg.bbox_3d_head)
        self.track_graph = build_track_graph(self.cfg.track_graph)

    def forward_train(
        self,
        batch_inputs: List[List[InputSample]],
    ) -> LossesType:
        """Forward function for training."""
        key_inputs, ref_inputs = self.preprocess_inputs(batch_inputs)

        # feature extraction
        key_x = self.detector.extract_features(key_inputs)
        ref_x = [self.detector.extract_features(inp) for inp in ref_inputs]

        # proposal generation
        key_proposals, rpn_losses = self.detector.generate_proposals(
            key_inputs, key_x
        )
        with torch.no_grad():
            ref_proposals = [
                self.detector.generate_proposals(inp, x)[0]
                for inp, x in zip(ref_inputs, ref_x)
            ]

        # bbox head
        _, roi_losses = self.detector.generate_detections(
            key_inputs,
            key_x,
            key_proposals,
            compute_detections=False,
        )

        # 3d bbox head
        loss_bbox_3d = self.bbox_3d_head.forward_train(
            key_inputs, key_x, key_proposals
        )

        det_losses = {**rpn_losses, **roi_losses, **loss_bbox_3d}

        # track head
        track_losses, _ = self.similarity_head.forward_train(
            [key_inputs, *ref_inputs],
            [key_x, *ref_x],
            [key_proposals, *ref_proposals],
        )

        return {**det_losses, **track_losses}

    def forward_test(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Forward function during inference."""
        assert len(batch_inputs) == 1, "No reference views during test!"
        raw_inputs = [inp[0] for inp in batch_inputs]
        assert len(raw_inputs) == 1, "Currently only BS=1 supported!"
        inputs = self.detector.preprocess_inputs(raw_inputs)

        # init graph at begin of sequence
        frame_id = inputs.metadata[0].frameIndex
        if frame_id == 0:
            self.track_graph.reset()

        # Detector
        feat = self.detector.extract_features(inputs)
        proposals, _ = self.detector.generate_proposals(inputs, feat)

        detections, _ = self.detector.generate_detections(
            inputs, feat, proposals
        )
        assert detections is not None

        bbox_2d_preds, bbox_3d_preds, keep = self.bbox_3d_head.forward_test(
            inputs,
            feat,
            detections,
        )

        # similarity head
        embeddings = self.similarity_head.forward_test(
            inputs, feat, detections
        )
        embeddings = embeddings[0][keep]
        assert inputs.metadata[0].size is not None
        input_size = (
            inputs[0].metadata[0].size.width,
            inputs[0].metadata[0].size.height,
        )
        self.postprocess(
            input_size, inputs.images.image_sizes[0], bbox_2d_preds
        )

        # associate detections, update graph
        tracks, tracks_3d = self.track_graph(
            bbox_2d_preds,
            bbox_3d_preds,
            frame_id,
            embeddings,
            inputs.extrinsics.tensor[0],
        )

        return dict(detect=[bbox_2d_preds], track=[tracks])
