"""Quasi-dense instance similarity learning model."""

from typing import List, Tuple

import torch

from vis4d.struct import InputSample, LossesType, ModelOutput

from .base import BaseModel, BaseModelConfig, build_model
from .detect import BaseTwoStageDetector
from .track.graph import TrackGraphConfig, build_track_graph
from .track.similarity import SimilarityLearningConfig, build_similarity_head
from .track.utils import split_key_ref_inputs


class QDTrack(BaseModel):
    """QDTrack model - quasi-dense instance similarity learning."""

    def __init__(self, detection: BaseModel,
    similarity: SimilarityLearning,
    track_graph: TrackGraph) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg = QDTrackConfig(**cfg.dict())  # type: QDTrackConfig
        assert self.cfg.category_mapping is not None
        self.cfg.detection.category_mapping = self.cfg.category_mapping
        self.detector = build_model(self.cfg.detection)
        assert isinstance(self.detector, BaseTwoStageDetector)
        self.similarity_head = build_similarity_head(self.cfg.similarity)
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}
        self.with_mask = self.detector.with_mask

    def preprocess_inputs(
        self,
        batch_inputs: List[List[InputSample]],
    ) -> Tuple[InputSample, List[InputSample]]:
        """Prepare images from key / ref input samples."""
        # split into key / ref pairs NxM input --> key: N, ref: Nx(M-1)
        key_inputs, ref_inputs = split_key_ref_inputs(batch_inputs)

        # group by ref views by sequence: Nx(M-1) --> (M-1)xN
        ref_inputs = [
            [ref_inputs[j][i] for j in range(len(ref_inputs))]
            for i in range(len(ref_inputs[0]))
        ]

        key_inputs_batch = self.detector.preprocess_inputs(key_inputs)
        ref_inputs_batch = [
            self.detector.preprocess_inputs(inp) for inp in ref_inputs
        ]
        return key_inputs_batch, ref_inputs_batch

    def forward_train(
        self,
        batch_inputs: List[List[InputSample]],
    ) -> LossesType:
        """Forward function for training."""
        key_inputs, ref_inputs = self.preprocess_inputs(batch_inputs)

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
        """Compute model output during inference."""
        assert len(batch_inputs[0]) == 1, "No reference views during test!"
        raw_inputs = [inp[0] for inp in batch_inputs]
        assert len(raw_inputs) == 1, "Currently only BS=1 supported!"
        inputs = self.detector.preprocess_inputs(raw_inputs)

        # init graph at begin of sequence
        frame_id = inputs.metadata[0].frameIndex
        if frame_id == 0:
            self.track_graph.reset()

        # detector
        feat = self.detector.extract_features(inputs)
        proposals, _ = self.detector.generate_proposals(inputs, feat)
        detections, _, segmentations = self.detector.generate_detections(
            inputs, feat, proposals, compute_segmentations=self.with_mask
        )
        assert detections is not None
        if segmentations is None or len(segmentations) == 0:
            segmentations = [None]

        # from vis4d.vis.image import imshow_bboxes
        # imshow_bboxes(inputs.images.tensor[0], detections)

        # similarity head
        embeddings = self.similarity_head.forward_test(
            inputs, feat, detections
        )
        assert inputs.metadata[0].size is not None
        input_size = (
            inputs.metadata[0].size.width,
            inputs.metadata[0].size.height,
        )
        detections[0].postprocess(input_size, inputs.images.image_sizes[0])
        if segmentations[0] is not None:
            segmentations[0].postprocess(
                input_size, inputs.images.image_sizes[0], detections[0]
            )

        # associate detections, update graph
        tracks = self.track_graph(detections[0], frame_id, embeddings[0])

        detects = (
            detections[0].to(torch.device("cpu")).to_scalabel(self.cat_mapping)
        )
        tracks_ = tracks.to(torch.device("cpu")).to_scalabel(self.cat_mapping)
        outputs = dict(detect=[detects], track=[tracks_])
        # Temporary hack to align masks with boxes for MOTS support
        # Remove in refactor-api PR!
        if segmentations[0] is not None:  # pragma: no cover
            segms = (
                segmentations[0]
                .to(torch.device("cpu"))
                .to_scalabel(self.cat_mapping)
            )
            track_inds = torch.empty((0), dtype=torch.int)
            track_ids = torch.empty((0), dtype=torch.int, device=tracks.device)
            for track in tracks:
                for i, box in enumerate(detections[0]):
                    if torch.equal(track.boxes, box.boxes):
                        track_inds = torch.cat(
                            [track_inds, torch.LongTensor([i])]
                        )
                        track_ids = torch.cat([track_ids, track.track_ids])
                        break
            if len(track_inds) == 0:
                segm_tracks_ = []
            else:
                segm_tracks = segmentations[0][track_inds]
                segm_tracks.track_ids = track_ids
                segm_tracks_ = segm_tracks.to(torch.device("cpu")).to_scalabel(
                    self.cat_mapping
                )
            outputs.update(segment=[segms], seg_track=[segm_tracks_])
        return outputs
