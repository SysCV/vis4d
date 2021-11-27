"""Quasi-dense 3D Tracking model."""
from typing import List

import torch

from ..struct import (
    Boxes2D,
    Boxes3D,
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
)
from .base import BaseModelConfig
from .detect.roi_head import BaseRoIHead, BaseRoIHeadConfig, build_roi_head
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
        self.bbox_3d_head: BaseRoIHead[Boxes3D] = build_roi_head(
            self.cfg.bbox_3d_head
        )
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}

    def forward_train(
        self,
        batch_inputs: List[InputSample],
    ) -> LossesType:
        """Forward function for training."""
        key_inputs, ref_inputs = self.preprocess_inputs(batch_inputs)
        key_targets, ref_targets = key_inputs.targets, [
            x.targets for x in ref_inputs
        ]

        # feature extraction
        key_x = self.detector.extract_features(key_inputs)
        ref_x = [self.detector.extract_features(inp) for inp in ref_inputs]

        # proposal generation
        key_proposals, rpn_losses = self.detector.generate_proposals(
            key_inputs, key_x
        )

        # 3d bbox head
        loss_bbox_3d, _ = self.bbox_3d_head(
            key_inputs, key_proposals, key_x, key_targets
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
        track_losses, _ = self.similarity_head(
            [key_inputs, *ref_inputs],
            [key_proposals, *ref_proposals],
            [key_x, *ref_x],
            [key_targets, *ref_targets],
        )

        return {**det_losses, **track_losses}

    def forward_test(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Compute qd-3dt output during inference."""
        assert len(batch_inputs[0]) == 1, "Currently only BS = 1 supported!"

        # if there is more than one InputSample, we switch to multi-sensor:
        # 1st elem is group, rest are sensor frames
        group = batch_inputs[0].to(self.device)
        if len(batch_inputs) > 1:
            frames = InputSample.cat(batch_inputs[1:])
        else:
            frames = batch_inputs[0]

        # detector
        inputs = self.detector.preprocess_inputs(frames)
        feat = self.detector.extract_features(inputs)
        proposals, _ = self.detector.generate_proposals(inputs, feat)

        boxes2d_list, _, _ = self.detector.generate_detections(
            inputs, feat, proposals
        )
        assert boxes2d_list is not None

        # 3d head
        boxes3d_list = self.bbox_3d_head(inputs, boxes2d_list, feat)

        # similarity head
        embeddings_list = self.similarity_head(inputs, boxes2d_list, feat)

        for inp, boxes2d in zip(inputs, boxes2d_list):
            assert inp.metadata[0].size is not None
            input_size = (
                inp.metadata[0].size.width,
                inp.metadata[0].size.height,
            )
            boxes2d.postprocess(
                input_size,
                inp.images.image_sizes[0],
                self.detector.cfg.clip_bboxes_to_image,
            )

        boxes2d = Boxes2D.merge(boxes2d_list)

        for idx, boxes3d in enumerate(boxes3d_list):
            assert isinstance(boxes3d, Boxes3D)
            boxes3d.transform(
                group.extrinsics.inverse() @ inputs[idx].extrinsics
            )
        boxes3d = Boxes3D.merge(boxes3d_list)
        embeds = [torch.cat(embeddings_list)]

        boxes_2d = boxes2d.to(torch.device("cpu")).to_scalabel(
            self.cat_mapping
        )
        # associate detections, update graph
        predictions = LabelInstances([boxes2d], [boxes3d])
        tracks = self.track_graph(inputs[0], predictions, embeddings=embeds)

        tracks_2d = (
            tracks.boxes2d[0]
            .to(torch.device("cpu"))
            .to_scalabel(self.cat_mapping)
        )
        tracks_3d = (
            tracks.boxes3d[0]
            .to(torch.device("cpu"))
            .to_scalabel(self.cat_mapping)
        )
        return dict(detect=[boxes_2d], track=[tracks_2d], track_3d=[tracks_3d])
