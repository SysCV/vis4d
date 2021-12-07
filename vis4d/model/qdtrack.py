"""Quasi-dense instance similarity learning model."""
from typing import List, Tuple

import torch

from vis4d.struct import (
    Boxes2D,
    FeatureMaps,
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
)

from .base import BaseModel, BaseModelConfig, build_model
from .detect import BaseTwoStageDetector
from .track.graph import TrackGraphConfig, build_track_graph
from .track.similarity import SimilarityLearningConfig, build_similarity_head
from .track.utils import split_key_ref_inputs


class QDTrackConfig(BaseModelConfig):
    """Config for quasi-dense tracking model."""

    detection: BaseModelConfig
    similarity: SimilarityLearningConfig
    track_graph: TrackGraphConfig


class QDTrack(BaseModel):
    """QDTrack model - quasi-dense instance similarity learning."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg: QDTrackConfig = QDTrackConfig(**cfg.dict())
        assert self.cfg.category_mapping is not None
        self.cfg.detection.category_mapping = self.cfg.category_mapping
        self.detector: BaseTwoStageDetector = build_model(self.cfg.detection)
        assert isinstance(self.detector, BaseTwoStageDetector)
        self.similarity_head = build_similarity_head(self.cfg.similarity)
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}
        self.with_mask = self.detector.with_mask

    def _detect_and_track_losses(
        self,
        key_inputs: InputSample,
        ref_inputs: List[InputSample],
        key_x: FeatureMaps,
        ref_x: List[FeatureMaps],
    ) -> Tuple[LossesType, List[Boxes2D], List[List[Boxes2D]]]:
        """Get detection and tracking losses."""
        key_targets, ref_targets = key_inputs.targets, [
            x.targets for x in ref_inputs
        ]

        # proposal generation
        rpn_losses, key_proposals = self.detector.generate_proposals(
            key_inputs, key_x, key_targets
        )
        with torch.no_grad():
            ref_proposals = [
                self.detector.generate_proposals(inp, x, tgt)[1]
                for inp, x, tgt in zip(ref_inputs, ref_x, ref_targets)
            ]

        # roi head
        roi_losses, _ = self.detector.generate_detections(
            key_inputs,
            key_x,
            key_proposals,
            key_targets,
        )
        det_losses = {**rpn_losses, **roi_losses}

        # from vis4d.vis.track import imshow_bboxes
        # for ref_imgs, ref_props in zip(ref_images, ref_proposals):
        #     for ref_img, ref_prop in zip(ref_imgs, ref_props):
        #         _, topk_i = torch.topk(ref_prop.boxes[:, -1], 100)
        #         imshow_bboxes(ref_img.tensor[0], ref_prop[topk_i])

        # track head
        track_losses, _ = self.similarity_head(
            [key_inputs, *ref_inputs],
            [key_proposals, *ref_proposals],
            [key_x, *ref_x],
            [key_targets, *ref_targets],
        )
        return {**det_losses, **track_losses}, key_proposals, ref_proposals

    def _detect_and_track(
        self, inputs: InputSample, feat: FeatureMaps
    ) -> ModelOutput:
        """Get detections and tracks."""
        proposals = self.detector.generate_proposals(inputs, feat)
        detections, instance_segms = self.detector.generate_detections(
            inputs, feat, proposals
        )

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
        if instance_segms is not None:
            instance_segms[0].postprocess(
                input_size, inputs.images.image_sizes[0], detections[0]
            )
            segms = (
                instance_segms[0]
                .to(torch.device("cpu"))
                .to_scalabel(self.cat_mapping)
            )
            outputs["ins_seg"] = [segms]

        # associate detections, update graph
        predictions = LabelInstances(
            detections,
            instance_masks=instance_segms
            if instance_segms is not None
            else None,
        )
        tracks = self.track_graph(inputs, predictions, embeddings=embeddings)

        tracks_ = (
            tracks.boxes2d[0]
            .to(torch.device("cpu"))
            .to_scalabel(self.cat_mapping)
        )
        outputs["track"] = [tracks_]

        if tracks.instance_masks[0] is not None:
            segm_tracks = (
                tracks.instance_masks[0]
                .to(torch.device("cpu"))
                .to_scalabel(self.cat_mapping)
            )
            outputs["seg_track"] = [segm_tracks]
        return outputs

    def forward_train(
        self,
        batch_inputs: List[InputSample],
    ) -> LossesType:
        """Forward function for training."""
        key_inputs, ref_inputs = split_key_ref_inputs(batch_inputs)

        # from vis4d.vis.image import imshow_bboxes
        # for batch_i, key_inp in enumerate(key_inputs):
        #     imshow_bboxes(key_inp.images.tensor[0], key_inp.boxes2d)
        #     for ref_i, ref_inp in enumerate(ref_inputs):
        #         imshow_bboxes(
        #             ref_inp[batch_i].images.tensor[0],
        #             ref_inp[batch_i].boxes2d,
        #         )

        # feature extraction
        key_x = self.detector.extract_features(key_inputs)
        ref_x = [self.detector.extract_features(inp) for inp in ref_inputs]

        losses, _, _ = self._detect_and_track_losses(
            key_inputs, ref_inputs, key_x, ref_x
        )
        return losses

    def forward_test(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Compute model output during inference."""
        assert len(batch_inputs) == 1, "No reference views during test!"
        assert len(batch_inputs[0]) == 1, "Currently only BS=1 supported!"
        feat = self.detector.extract_features(batch_inputs[0])
        return self._detect_and_track(batch_inputs[0], feat)
