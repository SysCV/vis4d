"""Quasi-dense instance similarity learning model."""
from typing import List

import torch

from vis4d.struct import InputSample, LabelInstances, LossesType, ModelOutput

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
        self.cfg = QDTrackConfig(**cfg.dict())  # type: QDTrackConfig
        assert self.cfg.category_mapping is not None
        self.cfg.detection.category_mapping = self.cfg.category_mapping
        self.detector: BaseTwoStageDetector = build_model(self.cfg.detection)
        assert isinstance(self.detector, BaseTwoStageDetector)
        self.similarity_head = build_similarity_head(self.cfg.similarity)
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}
        self.with_mask = self.detector.with_mask

    def preprocess_inputs(
        self, batch_inputs: List[InputSample]
    ) -> List[InputSample]:
        """Prepare images from key / ref input samples."""
        inputs_batch = [
            self.detector.preprocess_inputs(inp) for inp in batch_inputs
        ]
        return inputs_batch

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
        return {**det_losses, **track_losses}

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
        detections, _, segmentations = self.detector.generate_detections(
            inputs, feat, proposals, compute_segmentations=self.with_mask
        )
        assert detections is not None
        if segmentations is None or len(segmentations) == 0:
            segmentations = [None]  # type: ignore

        # from vis4d.vis.image import imshow_bboxes
        # imshow_bboxes(inputs.images.tensor[0], detections)

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
        if segmentations[0] is not None:
            segmentations[0].postprocess(
                input_size, inputs.images.image_sizes[0], detections[0]
            )
            segms = (
                segmentations[0]
                .to(torch.device("cpu"))
                .to_scalabel(self.cat_mapping)
            )
            outputs["segment"] = [segms]

        # associate detections, update graph
        predictions = LabelInstances(
            detections,
            instance_masks=segmentations
            if segmentations[0] is not None
            else None,
        )
        tracks = self.track_graph(inputs, predictions, embeddings=embeddings)

        tracks_ = (
            tracks.boxes2d[0]
            .to(torch.device("cpu"))
            .to_scalabel(self.cat_mapping)
        )
        outputs["track"] = [tracks_]

        if segmentations[0] is not None:
            segm_tracks = (
                tracks.instance_masks[0]
                .to(torch.device("cpu"))
                .to_scalabel(self.cat_mapping)
            )
            outputs["seg_track"] = [segm_tracks]
        return outputs
