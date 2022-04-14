"""Quasi-dense 3D Tracking model."""
from typing import List, Tuple, Union

import torch

from vis4d.common.module import build_module
from vis4d.struct import (
    ArgsType,
    Boxes2D,
    Boxes3D,
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
    ModuleCfg,
)

from .detect import BaseTwoStageDetector
from .heads.roi_head import BaseRoIHead, Det3DRoIHead
from .qdtrack import QDTrack
from .track.utils import split_key_ref_inputs


class QD3DT(QDTrack):
    """QD-3DT model class."""

    def __init__(
        self,
        bbox_3d_head: Union[Det3DRoIHead, ModuleCfg],
        *args: ArgsType,
        **kwargs: ArgsType
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        assert self.category_mapping is not None
        if isinstance(bbox_3d_head, dict):
            bbox_3d_head["num_classes"] = len(self.category_mapping)
            self.bbox_3d_head: Det3DRoIHead = build_module(
                bbox_3d_head, bound=BaseRoIHead
            )
        else:  # pragma: no cover
            self.bbox_3d_head = bbox_3d_head

    def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
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
        boxes2d, boxes3d, embeds = self.post_processing(
            boxes2d, boxes3d, embeds
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

    @staticmethod
    def post_processing(
        boxes2d: Boxes2D, boxes3d: Boxes3D, embeds: torch.Tensor
    ) -> Tuple[Boxes2D, Boxes3D, torch.Tensor]:
        """Post process the multi-camera results."""
        keep_indices = torch.ones(len(boxes3d))
        for i, box3d in enumerate(boxes3d):
            current_3d_score = box3d.score * boxes2d[i].score  # type: ignore
            if box3d.class_ids in [0, 1, 2, 8, 9]:
                nms_dist = 1
            else:
                nms_dist = 2
            distance = torch.cdist(boxes3d.center, box3d.center)
            nms_candidates = (distance < nms_dist).nonzero().squeeze(-1)
            for candiate in nms_candidates:
                if boxes3d[candiate].class_ids == box3d.class_ids and (
                    boxes2d[candiate].score * boxes3d[candiate].score  # type: ignore # pylint: disable=line-too-long
                    > current_3d_score
                ):
                    keep_indices[i] = 0
                    break
        keep = keep_indices == 1
        return boxes2d[keep], boxes3d[keep], embeds[keep]
