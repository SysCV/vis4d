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
        assert self.cfg.category_mapping is not None
        self.cfg.bbox_3d_head.num_classes = len(self.cfg.category_mapping)  # type: ignore # pylint: disable=line-too-long
        self.bbox_3d_head = build_roi_head(self.cfg.bbox_3d_head)
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}

    def forward_train(
        self,
        batch_inputs: List[InputSample],
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

        # 3d bbox head
        loss_bbox_3d, _ = self.bbox_3d_head.forward_train(
            key_inputs, key_proposals, key_x
        )

        # bbox head
        _, roi_losses, _ = self.detector.generate_detections(
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
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Compute qd-3dt output during inference."""
        assert len(batch_inputs[0]) == 1, "Currently only BS = 1 supported!"

        # if there is more than one InputSample per batch element, we switch
        # to multi-sensor mode: 1st elem is group, rest are sensor frames
        group = batch_inputs[0].to(self.device)
        if len(batch_inputs[0]) > 1:
            frames = batch_inputs[1:]
        else:
            frames = batch_inputs[0]

        # init graph at begin of sequence
        frame_id = group.metadata[0].frameIndex
        if frame_id == 0:
            self.track_graph.reset()

        # detector
        inputs = self.detector.preprocess_inputs(frames)
        feat = self.detector.extract_features(inputs)
        proposals, _ = self.detector.generate_proposals(inputs, feat)

        boxes2d_list, _, _ = self.detector.generate_detections(
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
            boxes2d.postprocess(input_size, inp.images.image_sizes[0])

        boxes2d = Boxes2D.merge(boxes2d_list)

        for idx, boxes3d in enumerate(boxes3d_list):
            assert isinstance(boxes3d, Boxes3D)
            boxes3d.transform(
                group.extrinsics.inverse() @ inputs[idx].extrinsics
            )
        boxes3d = Boxes3D.merge(boxes3d_list)  # type: ignore

        embeddings = torch.cat(embeddings_list)

        # associate detections, update graph
        tracks2d = self.track_graph(boxes2d, frame_id, embeddings)

        boxes_3d = torch.empty(
            (0, boxes3d.boxes.shape[1]), device=boxes3d.device
        )
        class_ids_3d = torch.empty((0), device=boxes3d.device)
        track_ids_3d = torch.empty((0), device=boxes3d.device)
        for track in tracks2d:
            for i, box in enumerate(boxes2d):
                if torch.equal(track.boxes, box.boxes):
                    if boxes3d[i].score is not None:
                        boxes3d[i].boxes[:, -1] *= track.score
                    boxes_3d = torch.cat([boxes_3d, boxes3d[i].boxes])
                    class_ids_3d = torch.cat([class_ids_3d, track.class_ids])
                    track_ids_3d = torch.cat([track_ids_3d, track.track_ids])

        # pylint: disable=no-member
        boxes_2d = boxes2d.to(torch.device("cpu")).to_scalabel(
            self.cat_mapping
        )
        tracks_2d = tracks2d.to(torch.device("cpu")).to_scalabel(
            self.cat_mapping
        )
        tracks_3d = (
            Boxes3D(boxes_3d, class_ids_3d, track_ids_3d)
            .to(torch.device("cpu"))
            .to_scalabel(self.cat_mapping)
        )
        return dict(detect=[boxes_2d], track=[tracks_2d], track_3d=[tracks_3d])
