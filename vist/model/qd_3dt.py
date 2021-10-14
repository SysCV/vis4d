"""Quasi-dense 3D Tracking model."""
from typing import List

import torch

from ..struct import Boxes2D, Boxes3D, InputSample, LossesType, ModelOutput
from .base import BaseModelConfig
from .detect.roi_head import BaseRoIHeadConfig, build_roi_head
from .qdtrack import QDTrack, QDTrackConfig
from .track.graph import build_track_graph


class QD3DTConfig(QDTrackConfig):
    """Config for quasi-dense 3D tracking model."""

    bbox_3d_head: BaseRoIHeadConfig


class QD3DT(QDTrack):
    """QD-3DT model class."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg = QD3DTConfig(**cfg.dict())
        self.cfg.bbox_3d_head.num_classes = len(self.cfg.category_mapping)  # type: ignore # pylint: disable=line-too-long
        self.bbox_3d_head = build_roi_head(self.cfg.bbox_3d_head)
        self.track_graph = build_track_graph(self.cfg.track_graph)

    def forward_train(
        self,
        batch_inputs: List[List[InputSample]],
    ) -> LossesType:
        """Forward function for training."""
        key_inputs, ref_inputs = self.preprocess_inputs(batch_inputs)

        # from vist.vis.image import imshow_bboxes3d
        # for batch_i, key_inp in enumerate(key_inputs):
        #     imshow_bboxes3d(key_inp.images.tensor[0], key_inp.boxes3d,
        #                     key_inp.intrinsics)
        #     for ref_i, ref_inp in enumerate(ref_inputs):
        #         imshow_bboxes3d(
        #             ref_inp[batch_i].images.tensor[0],
        #             ref_inp[batch_i].boxes3d,
        #             ref_inp[batch_i].intrinsics
        #         )

        # feature extraction
        key_x = self.detector.extract_features(key_inputs)
        ref_x = [self.detector.extract_features(inp) for inp in ref_inputs]

        # proposal generation
        key_proposals, rpn_losses = self.detector.generate_proposals(
            key_inputs, key_x
        )

        # 3d bbox head
        loss_bbox_3d, _ = self.bbox_3d_head.forward_train(
            key_inputs, key_proposals, key_x
        )

        # bbox head
        _, roi_losses = self.detector.generate_detections(
            key_inputs,
            key_x,
            key_proposals,
            compute_detections=False,
        )

        det_losses = {**rpn_losses, **roi_losses, **loss_bbox_3d}

        with torch.no_grad():
            ref_proposals = [
                self.detector.generate_proposals(inp, x)[0]
                for inp, x in zip(ref_inputs, ref_x)
            ]

        # track head
        track_losses, _ = self.similarity_head.forward_train(
            [key_inputs, *ref_inputs],
            [key_x, *ref_x],
            [key_proposals, *ref_proposals],
        )

        return {**det_losses, **track_losses}

    def forward_test(
        self,
        batch_inputs: List[List[InputSample]],
    ) -> ModelOutput:
        """Compute qd-3dt output during inference."""
        assert len(batch_inputs) == 1, "Currently only BS = 1 supported!"
        group, frames = batch_inputs[0][0], batch_inputs[0][1:]

        # init graph at begin of sequence
        frame_id = group.metadata[0].frameIndex
        if frame_id == 0:
            self.track_graph.reset()

        # detector
        inputs = self.detector.preprocess_inputs(frames)
        feat = self.detector.extract_features(inputs)
        proposals, _ = self.detector.generate_proposals(inputs, feat)

        boxes2d_list, _ = self.detector.generate_detections(
            inputs, feat, proposals
        )

        # 3d head
        boxes3d_list = self.bbox_3d_head.forward_test(
            inputs, boxes2d_list, feat
        )

        # similarity head
        embeddings_list = self.similarity_head.forward_test(
            inputs, feat, boxes2d_list
        )

        for inp, boxes2d in zip(inputs, boxes2d_list):
            assert inp.metadata[0].size is not None
            input_size = (
                inp.metadata[0].size.width,
                inp.metadata[0].size.height,
            )
            self.postprocess(input_size, inp.images.image_sizes[0], boxes2d)

        boxes2d = Boxes2D.merge(boxes2d_list)
        boxes3d = Boxes3D.merge(boxes3d_list)
        embeddings = torch.cat(embeddings_list)

        # associate detections, update graph
        tracks_2d = self.track_graph(boxes2d, frame_id, embeddings)

        boxes_3d = []
        for i in range(len(tracks_2d)):
            for j in range(len(boxes2d)):
                if torch.equal(tracks_2d.boxes[i], boxes2d.boxes[j]):
                    boxes_3d.append(boxes3d[j].boxes)

        boxes_3d = (
            torch.cat(boxes_3d)
            if len(boxes_3d) > 0
            else torch.empty(
                (0, boxes3d.boxes.shape[1]), device=boxes3d.device
            )
        )
        tracks_3d = Boxes3D(boxes_3d, tracks_2d.class_ids, tracks_2d.track_ids)

        return dict(detect=[boxes2d], track=[tracks_2d], track_3d=[tracks_3d])
