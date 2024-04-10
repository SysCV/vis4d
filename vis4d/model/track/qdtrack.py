"""Quasi-dense instance similarity learning model."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from vis4d.common.ckpt import load_model_checkpoint
from vis4d.model.detect.yolox import REV_KEYS as YOLOX_REV_KEYS
from vis4d.op.base import BaseModel, CSPDarknet, ResNet
from vis4d.op.box.box2d import scale_and_clip_boxes
from vis4d.op.box.encoder import DeltaXYWHBBoxDecoder
from vis4d.op.box.poolers import MultiScaleRoIAlign
from vis4d.op.detect.faster_rcnn import FasterRCNNHead, FRCNNOut
from vis4d.op.detect.rcnn import RoI2Det
from vis4d.op.detect.yolox import YOLOXHead, YOLOXOut, YOLOXPostprocess
from vis4d.op.fpp import FPN, YOLOXPAFPN, FeaturePyramidProcessing
from vis4d.op.track.common import TrackOut
from vis4d.op.track.qdtrack import (
    QDSimilarityHead,
    QDTrackAssociation,
    QDTrackHead,
)
from vis4d.state.track.qdtrack import QDTrackGraph

from .util import split_key_ref_indices

REV_KEYS = [
    (r"^faster_rcnn_heads\.", "faster_rcnn_head."),
    (r"^backbone.body\.", "basemodel."),
    (r"^qdtrack\.", "qdtrack_head."),
]


class FasterRCNNQDTrackOut(NamedTuple):
    """Output of QDtrack model."""

    detector_out: FRCNNOut
    key_images_hw: list[tuple[int, int]]
    key_target_boxes: list[Tensor]
    key_embeddings: list[Tensor]
    ref_embeddings: list[list[Tensor]]
    key_track_ids: list[Tensor]
    ref_track_ids: list[list[Tensor]]


class FasterRCNNQDTrack(nn.Module):
    """Wrap QDTrack with Faster R-CNN detector."""

    def __init__(
        self,
        num_classes: int,
        basemodel: BaseModel | None = None,
        faster_rcnn_head: FasterRCNNHead | None = None,
        rcnn_box_decoder: DeltaXYWHBBoxDecoder | None = None,
        qdtrack_head: QDTrackHead | None = None,
        track_graph: QDTrackGraph | None = None,
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
            qdtrack_head (QDTrack, optional): QDTrack head. Defaults to None.
                If None, will use default QDTrackHead.
            track_graph (QDTrackGraph, optional): Track graph. Defaults to
                None. If None, will use default QDTrackGraph.
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

        self.qdtrack_head = (
            QDTrackHead() if qdtrack_head is None else qdtrack_head
        )

        self.track_graph = (
            QDTrackGraph() if track_graph is None else track_graph
        )

        if weights is not None:
            load_model_checkpoint(
                self, weights, map_location="cpu", rev_keys=REV_KEYS
            )

    def forward(
        self,
        images: list[Tensor] | Tensor,
        images_hw: list[list[tuple[int, int]]] | list[tuple[int, int]],
        original_hw: list[list[tuple[int, int]]] | list[tuple[int, int]],
        frame_ids: list[list[int]] | list[int],
        boxes2d: None | list[list[Tensor]] = None,
        boxes2d_classes: None | list[list[Tensor]] = None,
        boxes2d_track_ids: None | list[list[Tensor]] = None,
        keyframes: None | list[list[bool]] = None,
    ) -> TrackOut | FasterRCNNQDTrackOut:
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
        return self._forward_test(images, images_hw, original_hw, frame_ids)  # type: ignore # pylint: disable=line-too-long

    def _forward_train(
        self,
        images: list[Tensor],
        images_hw: list[list[tuple[int, int]]],
        target_boxes: list[list[Tensor]],
        target_classes: list[list[Tensor]],
        target_track_ids: list[list[Tensor]],
        keyframes: list[list[bool]],
    ) -> FasterRCNNQDTrackOut:
        """Forward training stage.

        Args:
            images (list[Tensor]): Input images.
            images_hw (list[list[tuple[int, int]]]): Input image resolutions.
            target_boxes (list[list[Tensor]]): Bounding box labels.
            target_classes (list[list[Tensor]]): Class labels.
            target_track_ids (list[list[Tensor]]): Track IDs.
            keyframes (list[list[bool]]): Whether the frame is a keyframe.

        Returns:
            FasterRCNNQDTrackOut: Raw model outputs.
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
        ) = self.qdtrack_head(
            features=[key_features, *ref_features],
            det_boxes=[key_proposals, *ref_proposals],
            target_boxes=[key_target_boxes, *ref_target_boxes],
            target_track_ids=[key_target_track_ids, *ref_target_track_ids],
        )
        assert (
            ref_embeddings is not None
            and key_track_ids is not None
            and ref_track_ids is not None
        )

        return FasterRCNNQDTrackOut(
            detector_out=key_detector_out,
            key_images_hw=images_hw[key_index],
            key_target_boxes=key_target_boxes,
            key_embeddings=key_embeddings,
            ref_embeddings=ref_embeddings,
            key_track_ids=key_track_ids,
            ref_track_ids=ref_track_ids,
        )

    def _forward_test(
        self,
        images: Tensor,
        images_hw: list[tuple[int, int]],
        original_hw: list[tuple[int, int]],
        frame_ids: list[int],
    ) -> TrackOut:
        """Forward inference stage."""
        features = self.basemodel(images)
        features = self.fpn(features)
        detector_out = self.faster_rcnn_head(features, images_hw)

        boxes, scores, class_ids = self.roi2det(
            *detector_out.roi, detector_out.proposals.boxes, images_hw
        )
        embeddings, _, _, _ = self.qdtrack_head(features, boxes)

        tracks = self.track_graph(
            embeddings, boxes, scores, class_ids, frame_ids
        )

        for i, boxs in enumerate(tracks.boxes):
            tracks.boxes[i] = scale_and_clip_boxes(
                boxs, original_hw[i], images_hw[i]
            )
        return tracks

    def __call__(
        self,
        images: list[Tensor] | Tensor,
        images_hw: list[list[tuple[int, int]]] | list[tuple[int, int]],
        original_hw: list[tuple[int, int]],
        frame_ids: list[list[int]] | list[int],
        boxes2d: None | list[list[Tensor]] = None,
        boxes2d_classes: None | list[list[Tensor]] = None,
        boxes2d_track_ids: None | list[list[Tensor]] = None,
        keyframes: None | list[list[bool]] = None,
    ) -> TrackOut | FasterRCNNQDTrackOut:
        """Type definition for call implementation."""
        return self._call_impl(
            images,
            images_hw,
            original_hw,
            frame_ids,
            boxes2d,
            boxes2d_classes,
            boxes2d_track_ids,
            keyframes,
        )


class YOLOXQDTrackOut(NamedTuple):
    """Output of QDtrack YOLOX model."""

    detector_out: YOLOXOut
    key_images_hw: list[tuple[int, int]]
    key_target_boxes: list[Tensor]
    key_target_classes: list[Tensor]
    key_embeddings: list[Tensor]
    ref_embeddings: list[list[Tensor]]
    key_track_ids: list[Tensor]
    ref_track_ids: list[list[Tensor]]


class YOLOXQDTrack(nn.Module):
    """Wrap QDTrack with YOLOX detector."""

    def __init__(
        self,
        num_classes: int,
        basemodel: BaseModel | None = None,
        fpn: FeaturePyramidProcessing | None = None,
        yolox_head: YOLOXHead | None = None,
        train_postprocessor: YOLOXPostprocess | None = None,
        test_postprocessor: YOLOXPostprocess | None = None,
        qdtrack_head: QDTrackHead | None = None,
        track_graph: QDTrackGraph | None = None,
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
            train_postprocessor (YOLOXPostprocess, optional): Post processor
                for training. Defaults to None. If None, will use
                YOLOXPostprocess.
            test_postprocessor (YOLOXPostprocess, optional): Post processor
                for testing. Defaults to None. If None, will use
                YOLOXPostprocess.
            qdtrack_head (QDTrack, optional): QDTrack head. Defaults to None.
                If None, will use default QDTrackHead.
            track_graph (QDTrackGraph, optional): Track graph. Defaults to
                None. If None, will use default QDTrackGraph.
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
        self.train_postprocessor = (
            YOLOXPostprocess(
                self.yolox_head.point_generator,
                self.yolox_head.box_decoder,
                nms_threshold=0.7,
                score_thr=0.0,
                nms_pre=2000,
                max_per_img=1000,
            )
            if train_postprocessor is None
            else train_postprocessor
        )
        self.test_postprocessor = (
            YOLOXPostprocess(
                self.yolox_head.point_generator,
                self.yolox_head.box_decoder,
                nms_threshold=0.65,
                score_thr=0.1,
            )
            if test_postprocessor is None
            else test_postprocessor
        )

        self.qdtrack_head = (
            QDTrackHead(
                QDSimilarityHead(
                    MultiScaleRoIAlign(
                        resolution=[7, 7],
                        strides=[8, 16, 32],
                        sampling_ratio=0,
                    ),
                    in_dim=320,
                )
            )
            if qdtrack_head is None
            else qdtrack_head
        )

        self.track_graph = (
            QDTrackGraph(
                track=QDTrackAssociation(
                    init_score_thr=0.5, obj_score_thr=0.35
                )
            )
            if track_graph is None
            else track_graph
        )

        if weights is not None:
            load_model_checkpoint(
                self, weights, map_location="cpu", rev_keys=YOLOX_REV_KEYS
            )

    def forward(
        self,
        images: list[Tensor] | Tensor,
        images_hw: list[list[tuple[int, int]]] | list[tuple[int, int]],
        original_hw: list[list[tuple[int, int]]] | list[tuple[int, int]],
        frame_ids: list[list[int]] | list[int],
        boxes2d: None | list[list[Tensor]] = None,
        boxes2d_classes: None | list[list[Tensor]] = None,
        boxes2d_track_ids: None | list[list[Tensor]] = None,
        keyframes: None | list[list[bool]] = None,
    ) -> TrackOut | YOLOXQDTrackOut:
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
        return self._forward_test(images, images_hw, original_hw, frame_ids)  # type: ignore # pylint: disable=line-too-long

    def _forward_train(
        self,
        images: list[Tensor],
        images_hw: list[list[tuple[int, int]]],
        target_boxes: list[list[Tensor]],
        target_classes: list[list[Tensor]],
        target_track_ids: list[list[Tensor]],
        keyframes: list[list[bool]],
    ) -> YOLOXQDTrackOut:
        """Forward training stage.

        Args:
            images (list[Tensor]): Input images.
            images_hw (list[list[tuple[int, int]]]): Input image resolutions.
            target_boxes (list[list[Tensor]]): Bounding box labels.
            target_classes (list[list[Tensor]]): Class labels.
            target_track_ids (list[list[Tensor]]): Track IDs.
            keyframes (list[list[bool]]): Whether the frame is a keyframe.

        Returns:
            YOLOXQDTrackOut: Raw model outputs.
        """
        key_index, ref_indices = split_key_ref_indices(keyframes)

        # feature extraction
        key_features = self.fpn(self.basemodel(images[key_index].contiguous()))
        ref_features = [
            self.fpn(self.basemodel(images[ref_index].contiguous()))
            for ref_index in ref_indices
        ]

        key_detector_out = self.yolox_head(key_features[-3:])
        key_proposals, _, _ = self.train_postprocessor(
            cls_outs=key_detector_out.cls_score,
            reg_outs=key_detector_out.bbox_pred,
            obj_outs=key_detector_out.objectness,
            images_hw=images_hw[key_index],
        )

        with torch.no_grad():
            ref_detector_out = [
                self.yolox_head(ref_feat[-3:]) for ref_feat in ref_features
            ]
            ref_proposals = [
                self.train_postprocessor(
                    cls_outs=ref_out.cls_score,
                    reg_outs=ref_out.bbox_pred,
                    obj_outs=ref_out.objectness,
                    images_hw=images_hw[ref_index],
                )[0]
                for ref_index, ref_out in zip(ref_indices, ref_detector_out)
            ]

        key_target_boxes = target_boxes[key_index]
        ref_target_boxes = [
            target_boxes[ref_index] for ref_index in ref_indices
        ]
        key_target_classes = target_classes[key_index]
        key_target_track_ids = target_track_ids[key_index]
        ref_target_track_ids = [
            target_track_ids[ref_index] for ref_index in ref_indices
        ]

        (
            key_embeddings,
            ref_embeddings,
            key_track_ids,
            ref_track_ids,
        ) = self.qdtrack_head(
            features=[key_features, *ref_features],
            det_boxes=[key_proposals, *ref_proposals],
            target_boxes=[key_target_boxes, *ref_target_boxes],
            target_track_ids=[key_target_track_ids, *ref_target_track_ids],
        )
        assert (
            ref_embeddings is not None
            and key_track_ids is not None
            and ref_track_ids is not None
        )

        return YOLOXQDTrackOut(
            detector_out=key_detector_out,
            key_images_hw=images_hw[key_index],
            key_target_boxes=key_target_boxes,
            key_target_classes=key_target_classes,
            key_embeddings=key_embeddings,
            ref_embeddings=ref_embeddings,
            key_track_ids=key_track_ids,
            ref_track_ids=ref_track_ids,
        )

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
        boxes, scores, class_ids = self.test_postprocessor(
            cls_outs=outs.cls_score,
            reg_outs=outs.bbox_pred,
            obj_outs=outs.objectness,
            images_hw=images_hw,
        )

        embeddings, _, _, _ = self.qdtrack_head(features, boxes)

        tracks = self.track_graph(
            embeddings, boxes, scores, class_ids, frame_ids
        )

        for i, boxs in enumerate(tracks.boxes):
            tracks.boxes[i] = scale_and_clip_boxes(
                boxs, original_hw[i], images_hw[i]
            )
        return tracks

    def __call__(
        self,
        images: list[Tensor] | Tensor,
        images_hw: list[list[tuple[int, int]]] | list[tuple[int, int]],
        original_hw: list[list[tuple[int, int]]] | list[tuple[int, int]],
        frame_ids: list[list[int]] | list[int],
        boxes2d: None | list[list[Tensor]] = None,
        boxes2d_classes: None | list[list[Tensor]] = None,
        boxes2d_track_ids: None | list[list[Tensor]] = None,
        keyframes: None | list[list[bool]] = None,
    ) -> TrackOut | FasterRCNNQDTrackOut:
        """Type definition for call implementation."""
        return self._call_impl(
            images,
            images_hw,
            original_hw,
            frame_ids,
            boxes2d,
            boxes2d_classes,
            boxes2d_track_ids,
            keyframes,
        )
