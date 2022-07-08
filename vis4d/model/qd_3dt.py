"""Quasi-dense 3D Tracking model."""
from typing import List

import torch

from vis4d.common.bbox.utils import distance_3d_nms
from vis4d.struct import (
    ArgsType,
    Boxes2D,
    Boxes3D,
    InputSample,
    Losses,
    ModelOutput,
)

from .detect import BaseTwoStageDetector
from .heads.roi_head import Det3DRoIHead
from .qdtrack import QDTrack
from .track.utils import split_key_ref_inputs


class QD3DT(QDTrack):
    """QD-3DT model class."""

    def __init__(
        self, bbox_3d_head: Det3DRoIHead, *args: ArgsType, **kwargs: ArgsType
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        assert self.category_mapping is not None
        self.bbox_3d_head = bbox_3d_head

    def forward_train(self, batch_inputs: List[InputSample]) -> Losses:
        """Forward function for training."""
        key_inputs, ref_inputs = split_key_ref_inputs(batch_inputs)
        key_targets = key_inputs.targets

        # feature extraction
        key_x = self.detector.extract_features(key_inputs)
        ref_x = [self.detector.extract_features(inp) for inp in ref_inputs]

        losses, key_proposals, _ = self._run_heads_train(
            key_inputs, ref_inputs, key_x, ref_x
        )

        # 3d bbox head
        loss_bbox_3d, _ = self.bbox_3d_head(
            key_inputs, key_x, key_proposals, key_targets
        )
        losses.update(loss_bbox_3d)
        return losses

    def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
        """Compute qd_3dt output during inference."""
        assert len(batch_inputs[0]) == 1, "Currently only BS = 1 supported!"

        # if there is more than one InputSample, we switch to multi-sensor:
        # 1st elem is group, rest are sensor frames
        group = batch_inputs[0].to(self.device)
        frames = (
            InputSample.cat(batch_inputs[1:])
            if len(batch_inputs) > 1
            else batch_inputs[0]
        )

        # detector
        assert isinstance(self.detector, BaseTwoStageDetector)
        feat = self.detector.extract_features(frames)
        proposals = self.detector.generate_proposals(frames, feat)

        boxes2d_list, _ = self.detector.generate_detections(
            frames, feat, proposals
        )

        # 3d head
        boxes3d_list = self.bbox_3d_head(frames, feat, boxes2d_list)

        # similarity head
        embeddings_list = self.similarity_head(frames, boxes2d_list, feat)

        for inp, boxes2d in zip(frames, boxes2d_list):
            assert inp.metadata[0].size is not None
            input_size = (
                inp.metadata[0].size.width,
                inp.metadata[0].size.height,
            )
            boxes2d.postprocess(
                input_size,
                inp.images.image_sizes[0],
                self.detector.clip_bboxes_to_image,
            )

        boxes2d = Boxes2D.merge(boxes2d_list)

        if sum(len(b) for b in boxes3d_list) == 0:  # pragma: no cover
            boxes3d = Boxes3D.merge(boxes3d_list)
        else:
            non_empty_3d_list = []
            for idx, boxes3d in enumerate(boxes3d_list):
                assert isinstance(boxes3d, Boxes3D)
                if len(boxes3d) != 0:
                    boxes3d.transform(frames[idx].extrinsics)
                    non_empty_3d_list.append(boxes3d)
            boxes3d = Boxes3D.merge(non_empty_3d_list)

        embeds = torch.cat(embeddings_list)

        # post processing
        keep_indices = distance_3d_nms(boxes3d, self.cat_mapping, boxes2d)
        boxes2d, boxes3d, embeds = (
            boxes2d[keep_indices],
            boxes3d[keep_indices],
            embeds[keep_indices],
        )

        # associate detections, update graph
        predictions = LabelInstances([boxes2d], [boxes3d])
        tracks = self.track_graph(frames[0], predictions, embeddings=[embeds])

        # Update 3D score and move 3D boxes into group sensor coordinate
        tracks.boxes3d[0].boxes[:, -1] = (
            tracks.boxes3d[0].score * tracks.boxes2d[0].score  # type: ignore
        )
        tracks.boxes3d[0].transform(group.extrinsics.inverse())

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
        return dict(
            detect=[tracks_2d],
            track=[tracks_2d],
            detect_3d=[tracks_3d],
            track_3d=[tracks_3d],
        )
