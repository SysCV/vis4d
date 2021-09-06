"""Quasi-dense instance similarity learning model."""

from typing import List, Tuple

import torch

from vist.struct import Boxes2D, Images, InputSample, LossesType, ModelOutput

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
        self.cfg.detection.category_mapping = self.cfg.category_mapping
        self.detector = build_model(self.cfg.detection)
        assert isinstance(self.detector, BaseTwoStageDetector)
        self.similarity_head = build_similarity_head(self.cfg.similarity)
        self.track_graph = build_track_graph(self.cfg.track_graph)

    def prepare_targets(
        self,
        key_inputs: List[InputSample],
        ref_inputs: List[List[InputSample]],
    ) -> Tuple[List[Boxes2D], List[List[Boxes2D]]]:
        """Prepare targets from key / ref input samples."""
        key_targets = []
        for x in key_inputs:
            assert x.boxes2d is not None
            key_targets.append(x.boxes2d.to(self.device))
        ref_targets = []
        for inputs in ref_inputs:
            ref_target = []
            for x in inputs:
                assert x.boxes2d is not None
                ref_target.append(x.boxes2d.to(self.device))
            ref_targets.append(ref_target)

        return key_targets, ref_targets

    def prepare_images(
        self,
        key_inputs: List[InputSample],
        ref_inputs: List[List[InputSample]],
    ) -> Tuple[Images, List[Images]]:
        """Prepare images from key / ref input samples."""
        key_images = self.detector.preprocess_image(key_inputs)
        ref_images = [
            self.detector.preprocess_image(inp) for inp in ref_inputs
        ]
        return key_images, ref_images

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
        key_targets, ref_targets = self.prepare_targets(key_inputs, ref_inputs)

        # from vist.vis.image import imshow_bboxes
        # for batch_i, key_inp in enumerate(key_inputs):
        #     imshow_bboxes(key_inp.image.tensor[0], key_targets[batch_i])
        #     for ref_i, ref_inp in enumerate(ref_inputs):
        #         imshow_bboxes(
        #             ref_inp[batch_i].image.tensor[0],
        #             ref_targets[ref_i][batch_i],
        #         )

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
        det_losses = {**rpn_losses, **roi_losses}

        # from vist.vis.track import imshow_bboxes
        # for ref_imgs, ref_props in zip(ref_images, ref_proposals):
        #     for ref_img, ref_prop in zip(ref_imgs, ref_props):
        #         _, topk_i = torch.topk(ref_prop.boxes[:, -1], 100)
        #         imshow_bboxes(ref_img.tensor[0], ref_prop[topk_i])

        # track head
        track_losses, _ = self.similarity_head.forward_train(
            [key_x, *ref_x],
            [key_proposals, *ref_proposals],
            [key_targets, *ref_targets],
        )
        return {**det_losses, **track_losses}

    def forward_test(
        self,
        batch_inputs: List[List[InputSample]],
    ) -> ModelOutput:
        """Compute model output during inference."""
        assert len(batch_inputs) == 1, "No reference views during test!"
        inputs = [inp[0] for inp in batch_inputs]
        assert len(inputs) == 1, "Currently only BS=1 supported!"

        # init graph at begin of sequence
        frame_id = inputs[0].metadata.frameIndex
        if frame_id == 0:
            self.track_graph.reset()

        # detector
        image = self.detector.preprocess_image(inputs)
        feat = self.detector.extract_features(image)
        proposals, _ = self.detector.generate_proposals(image, feat)
        detections, _ = self.detector.generate_detections(
            image, feat, proposals
        )
        assert detections is not None

        # from vist.vis.image import imshow_bboxes
        # imshow_bboxes(image.tensor[0], detections)

        # similarity head
        embeddings = self.similarity_head.forward_test(feat, detections)
        assert inputs[0].metadata.size is not None
        input_size = (
            inputs[0].metadata.size.width,
            inputs[0].metadata.size.height,
        )
        self.postprocess(input_size, image.image_sizes[0], detections[0])

        # associate detections, update graph
        tracks = self.track_graph(detections[0], frame_id, embeddings[0])
        return dict(detect=detections, track=[tracks])
