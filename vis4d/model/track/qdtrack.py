"""Quasi-dense instance similarity learning model."""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from vis4d.common.ckpt import load_model_checkpoint
from vis4d.op.base import BaseModel, CSPDarknet, ResNet
from vis4d.op.box.box2d import scale_and_clip_boxes
from vis4d.op.box.encoder import DeltaXYWHBBoxDecoder
from vis4d.op.box.matchers import MaxIoUMatcher
from vis4d.op.box.poolers import MultiScaleRoIAlign
from vis4d.op.box.samplers import CombinedSampler, match_and_sample_proposals
from vis4d.op.detect.faster_rcnn import FasterRCNNHead
from vis4d.op.detect.rcnn import RoI2Det
from vis4d.op.detect.yolox import YOLOXHead, YOLOXPostprocess
from vis4d.op.fpp import FPN, YOLOXPAFPN, FeaturePyramidProcessing
from vis4d.op.track.assignment import TrackIDCounter
from vis4d.op.track.qdtrack import (
    QDSimilarityHead,
    QDTrackAssociation,
    QDTrackInstanceSimilarityLoss,
    QDTrackInstanceSimilarityLosses,
)
from vis4d.state.track.qdtrack import QDTrackMemory, QDTrackState

REV_KEYS = [
    # (r"^detector.rpn_head.mm_dense_head\.", "rpn_head."),
    # (r"\.rpn_reg\.", ".rpn_box."),
    # (r"^detector.roi_head.mm_roi_head.bbox_head\.", "roi_head."),
    # (r"^detector.backbone.mm_backbone\.", "body."),
    # (
    #     r"^detector.backbone.neck.mm_neck.lateral_convs\.",
    #     "inner_blocks.",
    # ),
    # (
    #     r"^detector.backbone.neck.mm_neck.fpn_convs\.",
    #     "layer_blocks.",
    # ),
    # (r"\.conv.weight", ".weight"),
    # (r"\.conv.bias", ".bias"),
    (r"^backbone.body\.", "basemodel."),
]

# from old Vis4D checkpoint
YOLOX_REV_KEYS = [
    (r"^detector.backbone.mm_backbone\.", "basemodel."),
    (r"^detector.backbone.neck.mm_neck\.", "fpn."),
    (r"^detector.bbox_head.mm_dense_head\.", "yolox_head."),
    (r"^similarity_head\.", "qdtrack.similarity_head."),
    (r"\.bn\.", ".norm."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


class TrackOut(NamedTuple):
    """Output of track 3D model."""

    boxes: list[Tensor]  # (N, 4)
    class_ids: list[Tensor]
    scores: list[Tensor]
    track_ids: list[Tensor]


class QDTrack(nn.Module):
    """QDTrack - quasi-dense instance similarity learning."""

    def __init__(
        self,
        similarity_head: QDSimilarityHead | None = None,
        track_graph: QDTrackAssociation | None = None,
        memory_size: int = 10,
        memory_momentum: float = 0.8,
        num_ref_views: int = 1,
        proposal_append_gt: bool = True,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.num_ref_views = num_ref_views
        self.similarity_head = (
            QDSimilarityHead() if similarity_head is None else similarity_head
        )

        # only in inference
        assert 0 <= memory_momentum <= 1.0
        self.memo_momentum = memory_momentum
        self.track_graph = (
            QDTrackAssociation() if track_graph is None else track_graph
        )
        self.track_memory = QDTrackMemory(memory_limit=memory_size)

        self.box_sampler = CombinedSampler(
            batch_size=256,
            positive_fraction=0.5,
            pos_strategy="instance_balanced",
            neg_strategy="iou_balanced",
        )

        self.box_matcher = MaxIoUMatcher(
            thresholds=[0.3, 0.7],
            labels=[0, -1, 1],
            allow_low_quality_matches=False,
        )
        self.proposal_append_gt = proposal_append_gt
        self.track_loss = QDTrackInstanceSimilarityLoss()

    def forward(
        self,
        features: list[torch.Tensor],
        det_boxes: list[torch.Tensor],
        det_scores: list[torch.Tensor],
        det_class_ids: list[torch.Tensor],
        frame_ids: None | tuple[int, ...] = None,
        target_boxes: None | list[torch.Tensor] = None,
        target_track_ids: None | list[torch.Tensor] = None,
    ) -> TrackOut | QDTrackInstanceSimilarityLosses:
        """Forward function."""
        if target_boxes is not None:
            assert (
                target_track_ids is not None
            ), "Need targets during training!"
            return self._forward_train(
                features,
                det_boxes,
                target_boxes,
                target_track_ids,
            )
        assert frame_ids is not None, "Need frame ids during inference!"
        return self._forward_test(
            features, det_boxes, det_scores, det_class_ids, frame_ids
        )

    def _split_views(
        self,
        embeddings: list[torch.Tensor],
        target_track_ids: list[torch.Tensor],
    ) -> tuple[
        list[torch.Tensor],
        list[list[torch.Tensor]],
        list[torch.Tensor],
        list[list[torch.Tensor]],
    ]:
        """Split batch and reference view dimension."""
        batch_size, ref_views = len(embeddings), self.num_ref_views + 1
        key_embeddings = [
            embeddings[i] for i in range(0, batch_size, ref_views)
        ]
        key_track_ids = [
            target_track_ids[i] for i in range(0, batch_size, ref_views)
        ]
        ref_embeddings, ref_track_ids = [], []
        for i in range(1, batch_size, ref_views):
            current_refs, current_track_ids = [], []
            for j in range(i, i + ref_views - 1):
                current_refs.append(embeddings[j])
                current_track_ids.append(target_track_ids[j])
            ref_embeddings.append(current_refs)
            ref_track_ids.append(current_track_ids)
        return key_embeddings, ref_embeddings, key_track_ids, ref_track_ids

    @torch.no_grad()
    def _sample_proposals(
        self,
        det_boxes: list[torch.Tensor],
        target_boxes: list[torch.Tensor],
        target_track_ids: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Sample proposals for instance similarity learning."""
        batch_size, ref_views = len(det_boxes), self.num_ref_views + 1

        if self.proposal_append_gt:
            det_boxes = [
                torch.cat([d, t]) for d, t in zip(det_boxes, target_boxes)
            ]

        (
            sampled_box_indices,
            sampled_target_indices,
            sampled_labels,
        ) = match_and_sample_proposals(
            self.box_matcher,
            self.box_sampler,
            det_boxes,
            target_boxes,
        )
        sampled_boxes, sampled_track_ids = [], []
        for i in range(batch_size):
            positives = sampled_labels[i] == 1
            if i % ref_views == 0:  # key view: take only positives
                sampled_box = det_boxes[i][sampled_box_indices[i]][positives]
                sampled_tr_id = target_track_ids[i][sampled_target_indices[i]][
                    positives
                ]
            else:  # set track_ids to -1 for all negatives
                sampled_box = det_boxes[i][sampled_box_indices[i]]
                sampled_tr_id = target_track_ids[i][sampled_target_indices[i]]
                sampled_tr_id[~positives] = -1

            sampled_boxes.append(sampled_box)
            sampled_track_ids.append(sampled_tr_id)
        return sampled_boxes, sampled_track_ids

    def _forward_train(
        self,
        features: list[torch.Tensor],
        det_boxes: list[torch.Tensor],
        target_boxes: list[torch.Tensor],
        target_track_ids: list[torch.Tensor],
    ) -> QDTrackInstanceSimilarityLosses:
        """Forward train."""  # TODO doc, verify training
        sampled_boxes, sampled_track_ids = self._sample_proposals(
            det_boxes, target_boxes, target_track_ids
        )
        embeddings = self.similarity_head(features, sampled_boxes)
        return self.track_loss(
            *self._split_views(embeddings, sampled_track_ids)
        )

    def _forward_test(
        self,
        features: list[torch.Tensor],
        det_boxes: list[torch.Tensor],
        det_scores: list[torch.Tensor],
        det_class_ids: list[torch.Tensor],
        frame_ids: tuple[int, ...],
    ) -> TrackOut:
        """Forward during test."""
        embeddings = self.similarity_head(features, det_boxes)

        batched_tracks = []
        for frame_id, box, score, cls_id, embeds in zip(
            frame_ids, det_boxes, det_scores, det_class_ids, embeddings
        ):
            # reset graph at begin of sequence
            if frame_id == 0:
                self.track_memory.reset()
                TrackIDCounter.reset()

            cur_memory = self.track_memory.get_current_tracks(box.device)
            track_ids, match_ids, filter_indices = self.track_graph(
                box,
                score,
                cls_id,
                embeds,
                cur_memory.track_ids,
                cur_memory.class_ids,
                cur_memory.embeddings,
            )

            valid_embeds = embeds[filter_indices]

            for i, track_id in enumerate(track_ids):
                if track_id in match_ids:
                    track = self.track_memory.get_track(track_id)[-1]
                    valid_embeds[i] = (
                        (1 - self.memo_momentum) * track.embeddings
                        + self.memo_momentum * valid_embeds[i]
                    )

            data = QDTrackState(
                track_ids,
                box[filter_indices],
                score[filter_indices],
                cls_id[filter_indices],
                valid_embeds,
            )
            self.track_memory.update(data)
            batched_tracks.append(self.track_memory.frames[-1])

        return TrackOut(
            [t.boxes for t in batched_tracks],
            [t.class_ids for t in batched_tracks],
            [t.scores for t in batched_tracks],
            [t.track_ids for t in batched_tracks],
        )


class FasterRCNNQDTrack(nn.Module):
    """Wrap QDTrack with Faster R-CNN detector."""

    def __init__(
        self,
        num_classes: int,
        basemodel: BaseModel | None = None,
        faster_rcnn_head: FasterRCNNHead | None = None,
        rcnn_box_decoder: DeltaXYWHBBoxDecoder | None = None,
        weights: None | str = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            num_classes (int): Number of object categories.
            basemodel (BaseModel, optional): Base model network. Defaults to
                None. If None, will use ResNet50.
            faster_rcnn_head (FasterRCNNHead, optional): Faster RCNN head.
                Defaults to None. if None, will use default FasterRCNNHead.
            rcnn_box_decoder (DeltaXYWHBBoxDecoder, optional): Decoder for RCNN
                bounding boxes. Defaults to None.
            weights (str, optional): Weights to load for model.
        """
        super().__init__()
        self.basemodel = (
            ResNet(resnet_name="resnet50", pretrained=True, trainable_layers=3)
            if basemodel is None
            else basemodel
        )

        self.fpn = FPN(self.basemodel.out_channels[2:], 256)

        if faster_rcnn_head is None:
            self.faster_rcnn_heads = FasterRCNNHead(num_classes=num_classes)
        else:
            self.faster_rcnn_heads = faster_rcnn_head

        self.roi2det = RoI2Det(rcnn_box_decoder)

        self.qdtrack = QDTrack()

        if weights is not None:
            load_model_checkpoint(
                self, weights, map_location="cpu", rev_keys=REV_KEYS
            )

    def forward(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        frame_ids: list[int],
    ) -> TrackOut:
        """Forward."""
        # TODO implement forward_train
        return self._forward_test(images, images_hw, frame_ids)

    def _forward_test(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        frame_ids: list[int],
    ) -> TrackOut:
        """Forward inference stage."""
        features = self.basemodel(images)
        features = self.fpn(features)
        detector_out = self.faster_rcnn_heads(features, images_hw)

        boxes, scores, class_ids = self.roi2det(
            *detector_out.roi, detector_out.proposals.boxes, images_hw
        )
        outs = self.qdtrack(features, boxes, scores, class_ids, frame_ids)
        return outs

    def __call__(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        frame_ids: list[int],
    ) -> TrackOut:
        """Type definition for call implementation."""
        return self._call_impl(images, images_hw, frame_ids)


class YOLOXQDTrack(nn.Module):
    """Wrap QDTrack with YOLOX detector."""

    def __init__(
        self,
        num_classes: int,
        basemodel: BaseModel | None = None,
        fpn: FeaturePyramidProcessing | None = None,
        yolox_head: YOLOXHead | None = None,
        weights: None | str = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            num_classes (int): Number of object categories.
            basemodel (BaseModel, optional): Base model. Defaults to None. If
                None, will use CSPDarknet.
            fpn (FeaturePyramidProcessing, optional): Feature Pyramid
                Processing. Defaults to None. If None, will use YOLOXPAFPN.
            yolox_head (YOLOXHead, optional): YOLOX head. Defaults to None. If
                None, will use YOLOXHead.
            weights (str, optional): Weights to load for model.
        """
        super().__init__()
        self.basemodel = (
            CSPDarknet(deepen_factor=1.33, widen_factor=1.25)
            if basemodel is None
            else basemodel
        )
        self.fpn = (
            YOLOXPAFPN([320, 640, 1280], 320, num_csp_blocks=4)
            if fpn is None
            else fpn
        )
        self.yolox_head = (
            YOLOXHead(
                num_classes=num_classes, in_channels=320, feat_channels=320
            )
            if yolox_head is None
            else yolox_head
        )
        self.transform_outs = YOLOXPostprocess(
            self.yolox_head.point_generator,
            self.yolox_head.box_decoder,
            nms_threshold=0.65,
            score_thr=0.1,
        )

        self.qdtrack = QDTrack(
            similarity_head=QDSimilarityHead(
                MultiScaleRoIAlign(
                    resolution=[7, 7], strides=[8, 16, 32], sampling_ratio=0
                ),
                in_dim=320,
            ),
            track_graph=QDTrackAssociation(
                init_score_thr=0.5, obj_score_thr=0.35
            ),
        )

        if weights is not None:
            load_model_checkpoint(
                self, weights, map_location="cpu", rev_keys=YOLOX_REV_KEYS
            )

    def forward(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        original_hw: list[tuple[int, int]],
        frame_ids: list[int],
    ) -> TrackOut:
        """Forward."""
        # TODO implement forward_train
        return self._forward_test(images, images_hw, original_hw, frame_ids)

    def _forward_test(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        original_hw: list[tuple[int, int]],
        frame_ids: list[int],
    ) -> TrackOut:
        """Forward inference stage."""
        features = self.fpn(self.basemodel(images))
        outs = self.yolox_head(features[-3:])
        boxes, scores, class_ids = self.transform_outs(
            cls_outs=outs.cls_score,
            reg_outs=outs.bbox_pred,
            obj_outs=outs.objectness,
            images_hw=images_hw,
        )

        tracks = self.qdtrack(features, boxes, scores, class_ids, frame_ids)
        for i, boxs in enumerate(tracks.boxes):
            tracks.boxes[i] = scale_and_clip_boxes(
                boxs, original_hw[i], images_hw[i]
            )
        return tracks

    def __call__(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        original_hw: list[tuple[int, int]],
        frame_ids: list[int],
    ) -> TrackOut:
        """Type definition for call implementation."""
        return self._call_impl(images, images_hw, original_hw, frame_ids)
