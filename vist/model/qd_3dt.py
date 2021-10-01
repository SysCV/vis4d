<<<<<<< HEAD
=======
# pylint: disable=duplicate-code
>>>>>>> main
"""Quasi-dense 3D Tracking model."""
from typing import List

import torch

<<<<<<< HEAD
from vist.struct import InputSample, LossesType, ModelOutput

from .detect.roi_head import BaseRoIHeadConfig, build_roi_head
from .qdtrack import QDTrack, QDTrackConfig
from .track.graph import build_track_graph
import pdb
=======
from vist.struct import Boxes3D, InputSample, LossesType, ModelOutput

from .base import BaseModelConfig
from .detect.roi_head import BaseRoIHeadConfig, build_roi_head
from .qdtrack import QDTrack, QDTrackConfig
from .track.graph import build_track_graph
>>>>>>> main


class QD3DTConfig(QDTrackConfig):
    """Config for quasi-dense 3D tracking model."""

    bbox_3d_head: BaseRoIHeadConfig


class QD3DT(QDTrack):
    """QD-3DT model class."""

<<<<<<< HEAD
    def __init__(self, cfg) -> None:
=======
    def __init__(self, cfg: BaseModelConfig) -> None:
>>>>>>> main
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
<<<<<<< HEAD
        with torch.no_grad():
            ref_proposals = [
                self.detector.generate_proposals(inp, x)[0]
                for inp, x in zip(ref_inputs, ref_x)
            ]
=======

        # 3d bbox head
        loss_bbox_3d, _ = self.bbox_3d_head.forward_train(
            key_inputs, key_x, key_proposals
        )
>>>>>>> main

        # bbox head
        _, roi_losses = self.detector.generate_detections(
            key_inputs,
            key_x,
            key_proposals,
            compute_detections=False,
        )

<<<<<<< HEAD
        # 3d bbox head
        loss_bbox_3d = self.bbox_3d_head.forward_train(
            key_inputs, key_x, key_proposals
        )

        det_losses = {**rpn_losses, **roi_losses, **loss_bbox_3d}

=======
        det_losses = {**rpn_losses, **roi_losses, **loss_bbox_3d}

        with torch.no_grad():
            ref_proposals = [
                self.detector.generate_proposals(inp, x)[0]
                for inp, x in zip(ref_inputs, ref_x)
            ]

>>>>>>> main
        # track head
        track_losses, _ = self.similarity_head.forward_train(
            [key_inputs, *ref_inputs],
            [key_x, *ref_x],
            [key_proposals, *ref_proposals],
        )

        return {**det_losses, **track_losses}

    def forward_test(
        self,
<<<<<<< HEAD
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Forward function during inference."""
        assert len(batch_inputs) == 1, "No reference views during test!"
        raw_inputs = [inp[0] for inp in batch_inputs]
        assert len(raw_inputs) == 1, "Currently only BS=1 supported!"
=======
        batch_inputs: List[List[InputSample]],
    ) -> ModelOutput:
        """Compute qd-3dt output during inference."""
        assert len(batch_inputs) == 1, "No reference views during test!"
        raw_inputs = [inp[0] for inp in batch_inputs]
        assert len(raw_inputs) == 1, "Currently only BS = 1 supported!"
>>>>>>> main
        inputs = self.detector.preprocess_inputs(raw_inputs)

        # init graph at begin of sequence
        frame_id = inputs.metadata[0].frameIndex
        if frame_id == 0:
            self.track_graph.reset()

        # Detector
        feat = self.detector.extract_features(inputs)
        proposals, _ = self.detector.generate_proposals(inputs, feat)

<<<<<<< HEAD
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
=======
        bbox_2d_preds, _ = self.detector.generate_detections(
            inputs, feat, proposals
        )
        assert bbox_2d_preds is not None

        bbox_3d_preds = self.bbox_3d_head.forward_test(
            inputs,
            feat,
            bbox_2d_preds,
        )[0]
        assert isinstance(bbox_3d_preds, Boxes3D)

        # similarity head
        embeddings = self.similarity_head.forward_test(
            inputs, feat, bbox_2d_preds
        )
>>>>>>> main
        assert inputs.metadata[0].size is not None
        input_size = (
            inputs[0].metadata[0].size.width,
            inputs[0].metadata[0].size.height,
        )
        self.postprocess(
<<<<<<< HEAD
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
=======
            input_size, inputs.images.image_sizes[0], bbox_2d_preds[0]
        )

        # associate detections, update graph
        tracks_2d = self.track_graph(bbox_2d_preds[0], frame_id, embeddings[0])

        boxes_3d = []
        for i in range(len(tracks_2d)):
            for j in range(len(bbox_2d_preds[0])):
                if torch.equal(tracks_2d.boxes[i], bbox_2d_preds[0].boxes[j]):
                    boxes_3d.append(bbox_3d_preds[j].boxes)

        boxes_3d = (
            torch.cat(boxes_3d)
            if len(boxes_3d) > 0
            else torch.empty(
                (0, bbox_3d_preds.boxes.shape[1]),
                device=bbox_3d_preds.device,
            )
        )

        tracks_3d = Boxes3D(boxes_3d, tracks_2d.class_ids, tracks_2d.track_ids)

        return dict(
            detect=bbox_2d_preds, track=[tracks_2d], track_3d=[tracks_3d]
        )
>>>>>>> main
