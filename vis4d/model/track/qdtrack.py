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
from vis4d.op.detect.faster_rcnn import FasterRCNNHead, FRCNNOut
from vis4d.op.detect.rcnn import RoI2Det
from vis4d.op.detect.yolox import YOLOXHead, YOLOXPostprocess
from vis4d.op.fpp import FPN, YOLOXPAFPN, FeaturePyramidProcessing
from vis4d.op.track.assignment import TrackIDCounter
from vis4d.op.track.qdtrack import QDSimilarityHead, QDTrackAssociation
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
    # (r"\.conv.weight", ".weigh2t"),
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
    """Output of track model."""

    boxes: list[Tensor]  # (N, 4)
    class_ids: list[Tensor]
    scores: list[Tensor]
    track_ids: list[Tensor]


class QDTrackOut(NamedTuple):
    """Output of QDtrack model."""

    detector_out: FRCNNOut
    key_images_hw: list[tuple[int, int]]
    key_target_boxes: list[Tensor]
    key_embeddings: list[Tensor]
    ref_embeddings: list[list[Tensor]]
    key_track_ids: list[Tensor]
    ref_track_ids: list[list[Tensor]]


def split_key_ref_indices(
    keyframes: list[list[bool]],
) -> tuple[int, list[int]]:
    """Get key frame from list of sample attributes."""
    key_ind = None
    ref_inds = []
    for i, is_keys in enumerate(keyframes):
        assert all(
            is_keys[0] == is_key for is_key in is_keys
        ), "Same batch should have the same view."
        if is_keys[0]:
            key_ind = i
        else:
            ref_inds.append(i)

    assert key_ind is not None, "Key frame not found."
    assert len(ref_inds) > 0, "No reference frames found."

    return key_ind, ref_inds


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

    def forward(
        self,
        features: list[Tensor] | list[list[Tensor]],
        det_boxes: list[Tensor] | list[list[Tensor]],
        det_scores: None | list[Tensor] = None,
        det_class_ids: None | list[Tensor] = None,
        frame_ids: None | list[int] = None,
        target_boxes: None | list[list[Tensor]] = None,
        target_track_ids: None | list[list[Tensor]] = None,
    ) -> (
        TrackOut
        | tuple[
            list[Tensor], list[list[Tensor]], list[Tensor], list[list[Tensor]]
        ]
    ):
        """Forward function."""
        if target_boxes is not None:
            assert (
                target_boxes is not None and target_track_ids is not None
            ), "Need targets during training!"
            return self._forward_train(
                features, det_boxes, target_boxes, target_track_ids  # type: ignore # pylint: disable=line-too-long
            )
        assert (
            frame_ids is not None
            and det_scores is not None
            and det_class_ids is not None
        )
        return self._forward_test(
            features, det_boxes, det_scores, det_class_ids, frame_ids  # type: ignore # pylint: disable=line-too-long
        )

    @torch.no_grad()
    def _sample_proposals(
        self,
        det_boxes: list[list[Tensor]],
        target_boxes: list[list[Tensor]],
        target_track_ids: list[list[Tensor]],
    ) -> tuple[list[list[Tensor]], list[list[Tensor]]]:
        """Sample proposals for instance similarity learning."""
        sampled_boxes, sampled_track_ids = [], []
        for i, (boxes, tgt_boxes) in enumerate(zip(det_boxes, target_boxes)):
            if self.proposal_append_gt:
                boxes = [torch.cat([d, t]) for d, t in zip(boxes, tgt_boxes)]

            (
                sampled_box_indices,
                sampled_target_indices,
                sampled_labels,
            ) = match_and_sample_proposals(
                self.box_matcher,
                self.box_sampler,
                boxes,
                tgt_boxes,
            )

            positives = [l == 1 for l in sampled_labels]
            if i == 0:  # key view: take only positives
                sampled_box = [
                    b[s_i][p]
                    for b, s_i, p in zip(boxes, sampled_box_indices, positives)
                ]
                sampled_tr_id = [
                    t[s_i][p]
                    for t, s_i, p in zip(
                        target_track_ids[i], sampled_target_indices, positives
                    )
                ]
            else:  # set track_ids to -1 for all negatives
                sampled_box = [
                    b[s_i] for b, s_i in zip(boxes, sampled_box_indices)
                ]
                sampled_tr_id = [
                    t[s_i]
                    for t, s_i in zip(
                        target_track_ids[i], sampled_target_indices
                    )
                ]
                for pos, samp_tgt in zip(positives, sampled_tr_id):
                    samp_tgt[~pos] = -1

            sampled_boxes.append(sampled_box)
            sampled_track_ids.append(sampled_tr_id)
        return sampled_boxes, sampled_track_ids

    def _forward_train(
        self,
        features: list[list[Tensor]],
        det_boxes: list[list[Tensor]],
        target_boxes: list[list[Tensor]],
        target_track_ids: list[list[Tensor]],
    ) -> tuple[
        list[Tensor], list[list[Tensor]], list[Tensor], list[list[Tensor]]
    ]:
        """Forward train."""  # TODO doc, verify training
        sampled_boxes, sampled_track_ids = self._sample_proposals(
            det_boxes, target_boxes, target_track_ids
        )

        embeddings = []
        for feats, boxes in zip(features, sampled_boxes):
            embeddings.append(self.similarity_head(feats, boxes))

        return (
            embeddings[0],
            embeddings[1:],
            sampled_track_ids[0],
            sampled_track_ids[1:],
        )

    def _forward_test(
        self,
        features: list[Tensor],
        det_boxes: list[Tensor],
        det_scores: list[Tensor],
        det_class_ids: list[Tensor],
        frame_ids: list[int],
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

    def __call__(
        self,
        features: list[Tensor] | list[list[Tensor]],
        det_boxes: list[Tensor] | list[list[Tensor]],
        det_scores: None | list[Tensor] = None,
        det_class_ids: None | list[Tensor] = None,
        frame_ids: None | list[int] = None,
        target_boxes: None | list[list[Tensor]] = None,
        target_track_ids: None | list[list[Tensor]] = None,
    ) -> (
        TrackOut
        | tuple[
            list[Tensor], list[list[Tensor]], list[Tensor], list[list[Tensor]]
        ]
    ):
        """Type definition for call implementation."""
        return self._call_impl(
            features,
            det_boxes,
            det_scores,
            det_class_ids,
            frame_ids,
            target_boxes,
            target_track_ids,
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
            self.faster_rcnn_head = FasterRCNNHead(num_classes=num_classes)
        else:
            self.faster_rcnn_head = faster_rcnn_head

        self.roi2det = RoI2Det(rcnn_box_decoder)

        self.qdtrack = QDTrack()

        if weights is not None:
            load_model_checkpoint(
                self, weights, map_location="cpu", rev_keys=REV_KEYS
            )

    def forward(
        self,
        images: list[Tensor] | Tensor,
        images_hw: list[list[tuple[int, int]]] | list[tuple[int, int]],
        frame_ids: list[list[int]] | list[int],
        boxes2d: None | list[list[Tensor]] = None,
        boxes2d_classes: None | list[list[Tensor]] = None,
        boxes2d_track_ids: None | list[list[Tensor]] = None,
        keyframes: None | list[list[bool]] = None,
    ) -> TrackOut | QDTrackOut:
        """Forward."""
        if self.training:
            assert (
                isinstance(images, list)
                and boxes2d is not None
                and boxes2d_classes is not None
                and boxes2d_track_ids is not None
                and keyframes is not None
            )
            return self._forward_train(
                images,
                images_hw,  # type: ignore
                boxes2d,
                boxes2d_classes,
                boxes2d_track_ids,
                keyframes,
            )
        return self._forward_test(images, images_hw, frame_ids)  # type: ignore

    def _forward_train(
        self,
        images: list[Tensor],
        images_hw: list[list[tuple[int, int]]],
        target_boxes: list[list[Tensor]],
        target_classes: list[list[Tensor]],
        target_track_ids: list[list[Tensor]],
        keyframes: list[list[bool]],
    ) -> QDTrackOut:
        """Forward training stage.

        Args:
            images (list[Tensor]): Input images.
            images_hw (list[list[tuple[int, int]]]): Input image resolutions.
            target_boxes (list[list[Tensor]]): Bounding box labels.
            target_classes (list[list[Tensor]]): Class labels.
            target_track_ids (list[list[Tensor]]): Track IDs.
            keyframes (list[list[bool]]): Whether the frame is a keyframe.

        Returns:
            FRCNNOut: Raw model outputs.
        """
        key_index, ref_indices = split_key_ref_indices(keyframes)

        # feature extraction
        key_features = self.fpn(self.basemodel(images[key_index]))
        ref_features = [
            self.fpn(self.basemodel(images[ref_index]))
            for ref_index in ref_indices
        ]

        key_detector_out = self.faster_rcnn_head(
            key_features,
            images_hw[key_index],
            target_boxes[key_index],
            target_classes[key_index],
        )

        with torch.no_grad():
            ref_detector_out = [
                self.faster_rcnn_head(
                    ref_features[i],
                    images_hw[ref_index],
                    target_boxes[ref_index],
                    target_classes[ref_index],
                )
                for i, ref_index in enumerate(ref_indices)
            ]

        key_proposals = key_detector_out.proposals.boxes
        ref_proposals = [ref.proposals.boxes for ref in ref_detector_out]
        key_target_boxes = target_boxes[key_index]
        ref_target_boxes = [
            target_boxes[ref_index] for ref_index in ref_indices
        ]
        key_target_track_ids = target_track_ids[key_index]
        ref_target_track_ids = [
            target_track_ids[ref_index] for ref_index in ref_indices
        ]

        (
            key_embeddings,
            ref_embeddings,
            key_track_ids,
            ref_track_ids,
        ) = self.qdtrack(
            features=[key_features, *ref_features],
            det_boxes=[key_proposals, *ref_proposals],
            target_boxes=[key_target_boxes, *ref_target_boxes],
            target_track_ids=[key_target_track_ids, *ref_target_track_ids],
        )

        return QDTrackOut(
            detector_out=key_detector_out,
            key_images_hw=images_hw[key_index],
            key_target_boxes=key_target_boxes,
            key_embeddings=key_embeddings,
            ref_embeddings=ref_embeddings,  # type: ignore
            key_track_ids=key_track_ids,
            ref_track_ids=ref_track_ids,  # type: ignore
        )

    def _forward_test(
        self,
        images: Tensor,
        images_hw: list[tuple[int, int]],
        frame_ids: list[int],
    ) -> TrackOut:
        """Forward inference stage."""
        features = self.basemodel(images)
        features = self.fpn(features)
        detector_out = self.faster_rcnn_head(features, images_hw)

        boxes, scores, class_ids = self.roi2det(
            *detector_out.roi, detector_out.proposals.boxes, images_hw
        )
        outs = self.qdtrack(features, boxes, scores, class_ids, frame_ids)
        return outs  # type: ignore

    def __call__(
        self,
        images: list[Tensor] | Tensor,
        images_hw: list[list[tuple[int, int]]] | list[tuple[int, int]],
        frame_ids: list[list[int]] | list[int],
        boxes2d: None | list[list[Tensor]] = None,
        boxes2d_classes: None | list[list[Tensor]] = None,
        boxes2d_track_ids: None | list[list[Tensor]] = None,
        keyframes: None | list[list[bool]] = None,
    ) -> TrackOut | QDTrackOut:
        """Type definition for call implementation."""
        return self._call_impl(
            images,
            images_hw,
            frame_ids,
            boxes2d,
            boxes2d_classes,
            boxes2d_track_ids,
            keyframes,
        )


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
        assert isinstance(tracks, TrackOut)
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
