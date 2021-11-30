"""Quasi-dense instance similarity learning model."""
from typing import List, Optional

import torch

from vis4d.struct import InputSample, LabelInstances, LossesType, ModelOutput

from .base import BaseModelConfig
from .heads.dense_head import (
    BaseDenseHeadConfig,
    MMDecodeHead,
    build_dense_head,
)
from .qdtrack import QDTrack, QDTrackConfig
from .track.utils import split_key_ref_inputs


class QDTrackSegConfig(QDTrackConfig):
    """Config for quasi-dense tracking model with segmentation head."""

    seg_head: Optional[BaseDenseHeadConfig] = None


class QDTrackSeg(QDTrack):
    """QDTrack model with segmentation head."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg: QDTrackSegConfig = QDTrackSegConfig(**cfg.dict())
        if self.cfg.seg_head is not None:
            self.cfg.seg_head.num_classes = len(self.cfg.category_mapping)  # type: ignore # pylint: disable=line-too-long
            self.seg_head: Optional[MMDecodeHead] = build_dense_head(
                self.cfg.seg_head
            )
        else:
            self.seg_head = None

    def forward_train(
        self,
        batch_inputs: List[InputSample],
    ) -> LossesType:
        """Forward function for training."""
        batch_inputs = self.preprocess_inputs(batch_inputs)
        key_inputs, ref_inputs = split_key_ref_inputs(batch_inputs)
        key_targets, ref_targets = key_inputs.targets, [
            x.targets for x in ref_inputs
        ]
        # feature extraction
        key_x = self.detector.extract_features(key_inputs)
        ref_x = [self.detector.extract_features(inp) for inp in ref_inputs]

        losses = {}
        if len(key_targets.boxes2d[0]) > 0:
            # proposal generation
            key_proposals, rpn_losses = self.detector.generate_proposals(
                key_inputs, key_x
            )
            with torch.no_grad():
                ref_proposals = [
                    self.detector.generate_proposals(inp, x)[0]
                    for inp, x in zip(ref_inputs, ref_x)
                ]

            # roi head
            _, roi_losses, _ = self.detector.generate_detections(
                key_inputs,
                key_x,
                key_proposals,
                compute_detections=False,
            )
            det_losses = {**rpn_losses, **roi_losses}

            # track head
            track_losses, _ = self.similarity_head(
                [key_inputs, *ref_inputs],
                [key_proposals, *ref_proposals],
                [key_x, *ref_x],
                [key_targets, *ref_targets],
            )

            losses.update(det_losses)
            losses.update(track_losses)

        if (
            len(key_targets.semantic_masks[0]) > 0
            and self.seg_head is not None
        ):
            # segmentation head
            seg_losses = self.seg_head(key_inputs, key_x, key_targets)
            losses.update(seg_losses)

        return losses

    def forward_test(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Compute model output during inference."""
        assert len(batch_inputs) == 1, "No reference views during test!"
        assert len(batch_inputs[0]) == 1, "Currently only BS=1 supported!"
        inputs = self.detector.preprocess_inputs(batch_inputs[0])

        # detector
        feat = self.detector.extract_features(inputs)
        proposals, _ = self.detector.generate_proposals(inputs, feat)
        detections, _, instance_segms = self.detector.generate_detections(
            inputs, feat, proposals, compute_segmentations=self.with_mask
        )
        assert detections is not None
        if instance_segms is None or len(instance_segms) == 0:
            instance_segms = [None]  # type: ignore

        # similarity head
        embeddings = self.similarity_head(inputs, detections, feat)
        assert inputs.metadata[0].size is not None
        input_size = (
            inputs.metadata[0].size.width,
            inputs.metadata[0].size.height,
        )
        detections[0].postprocess(
            input_size,
            inputs.images.image_sizes[0],
            self.detector.cfg.clip_bboxes_to_image,
        )
        detects = (
            detections[0].to(torch.device("cpu")).to_scalabel(self.cat_mapping)
        )
        outputs = dict(detect=[detects])
        if instance_segms[0] is not None:
            instance_segms[0].postprocess(
                input_size, inputs.images.image_sizes[0], detections[0]
            )
            instance_segms_ = (
                instance_segms[0]
                .to(torch.device("cpu"))
                .to_scalabel(self.cat_mapping)
            )
            outputs["ins_seg"] = [instance_segms_]

        # associate detections, update graph
        predictions = LabelInstances(
            detections,
            instance_masks=instance_segms
            if instance_segms[0] is not None
            else None,
        )
        tracks = self.track_graph(inputs, predictions, embeddings=embeddings)

        tracks_ = (
            tracks.boxes2d[0]
            .to(torch.device("cpu"))
            .to_scalabel(self.cat_mapping)
        )
        outputs["track"] = [tracks_]

        if instance_segms[0] is not None:
            segm_tracks = (
                tracks.instance_masks[0]
                .to(torch.device("cpu"))
                .to_scalabel(self.cat_mapping)
            )
            outputs["seg_track"] = [segm_tracks]

        # segmentation head
        if self.seg_head is not None:
            semantic_segms = self.seg_head(inputs, feat)
            semantic_segms_ = (
                semantic_segms[0]
                .to(torch.device("cpu"))
                .to_scalabel(self.cat_mapping)
            )
            outputs["sem_seg"] = [semantic_segms_]

        return outputs
