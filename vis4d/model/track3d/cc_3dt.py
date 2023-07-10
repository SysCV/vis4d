"""CC-3DT model implementation.

This file composes the operations associated with CC-3DT
`https://arxiv.org/abs/2212.01247'`_.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from vis4d.common.ckpt import load_model_checkpoint
from vis4d.model.track.qdtrack import FasterRCNNQDTrackOut
from vis4d.op.base import BaseModel, ResNet
from vis4d.op.box.anchor import AnchorGenerator
from vis4d.op.box.encoder import DeltaXYWHBBoxDecoder
from vis4d.op.detect3d.qd_3dt import QD3DTBBox3DHead, RoI2Det3D
from vis4d.op.detect.faster_rcnn import FasterRCNNHead
from vis4d.op.detect.rcnn import RCNNHead, RoI2Det
from vis4d.op.fpp import FPN
from vis4d.op.track3d.common import Track3DOut
from vis4d.op.track.qdtrack import QDTrack
from vis4d.state.track3d.cc_3dt import CC3DTrackGraph

from ..track.util import split_key_ref_indices

REV_KEYS = [
    (r"^backbone.body\.", "basemodel."),
    (r"^faster_rcnn_heads\.", "faster_rcnn_head."),
    (r"^track\.", "qdtrack."),
]


class FasterRCNNCC3DTOut(NamedTuple):
    """Output of CC-3DT model with Faster R-CNN detector."""

    detector_3d_out: Tensor
    detector_3d_target: Tensor
    detector_3d_labels: Tensor
    qdtrack_out: FasterRCNNQDTrackOut


class FasterRCNNCC3DT(nn.Module):
    """CC-3DT with Faster-RCNN detector."""

    def __init__(
        self,
        num_classes: int,
        basemodel: BaseModel | None = None,
        faster_rcnn_head: FasterRCNNHead | None = None,
        rcnn_box_decoder: DeltaXYWHBBoxDecoder | None = None,
        qdtrack: QDTrack | None = None,
        track_graph: CC3DTrackGraph | None = None,
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
            qdtrack (QDTrack, optional): QDTrack. Defaults to None. If None,
                will use default QDTrack.
            track_graph (CC3DTrackGraph, optional): Track graph. Defaults to
                None. If None, will use default CC3DTrackGraph.
            weights (None | str): Weights path. Defaults to None.
        """
        super().__init__()
        self.basemodel = (
            ResNet(resnet_name="resnet50", pretrained=True, trainable_layers=3)
            if basemodel is None
            else basemodel
        )

        self.fpn = FPN(self.basemodel.out_channels[2:], 256)

        if faster_rcnn_head is None:
            anchor_generator = AnchorGenerator(
                scales=[4, 8],
                ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
                strides=[4, 8, 16, 32, 64],
            )
            roi_head = RCNNHead(num_shared_convs=4, num_classes=num_classes)
            self.faster_rcnn_head = FasterRCNNHead(
                num_classes=num_classes,
                anchor_generator=anchor_generator,
                roi_head=roi_head,
            )
        else:
            self.faster_rcnn_head = faster_rcnn_head

        self.roi2det = RoI2Det(rcnn_box_decoder)

        self.bbox_3d_head = QD3DTBBox3DHead(num_classes=num_classes)

        self.roi2det_3d = RoI2Det3D()

        self.qdtrack = QDTrack() if qdtrack is None else qdtrack

        self.track_graph = (
            CC3DTrackGraph() if track_graph is None else track_graph
        )

        if weights is not None:
            load_model_checkpoint(
                self, weights, map_location="cpu", rev_keys=REV_KEYS
            )

    def forward(
        self,
        images: list[Tensor] | Tensor,
        images_hw: list[list[tuple[int, int]]],
        intrinsics: list[Tensor] | Tensor,
        extrinsics: Tensor | None = None,
        frame_ids: list[list[int]] | None = None,
        boxes2d: list[list[Tensor]] | None = None,
        boxes3d: list[list[Tensor]] | None = None,
        boxes3d_classes: list[list[Tensor]] | None = None,
        boxes3d_track_ids: list[list[Tensor]] | None = None,
        keyframes: None | list[list[bool]] | None = None,
    ) -> FasterRCNNCC3DTOut | Track3DOut:
        """Forward."""
        if self.training:
            assert (
                isinstance(images, list)
                and isinstance(intrinsics, list)
                and boxes2d is not None
                and boxes3d is not None
                and boxes3d_classes is not None
                and boxes3d_track_ids is not None
                and keyframes is not None
            )
            return self._forward_train(
                images,
                images_hw,
                intrinsics,
                boxes2d,
                boxes3d,
                boxes3d_classes,
                boxes3d_track_ids,
                keyframes,
            )

        assert (
            not isinstance(images, list)
            and not isinstance(intrinsics, list)
            and extrinsics is not None
            and frame_ids is not None
        )
        return self._forward_test(
            images, images_hw, intrinsics, extrinsics, frame_ids
        )

    def _forward_train(
        self,
        images: list[Tensor],
        images_hw: list[list[tuple[int, int]]],
        intrinsics: list[Tensor],
        target_boxes2d: list[list[Tensor]],
        target_boxes3d: list[list[Tensor]],
        target_classes: list[list[Tensor]],
        target_track_ids: list[list[Tensor]],
        keyframes: list[list[bool]],
    ) -> FasterRCNNCC3DTOut:
        """Foward training stage."""
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
            target_boxes2d[key_index],
            target_classes[key_index],
        )

        with torch.no_grad():
            ref_detector_out = [
                self.faster_rcnn_head(
                    ref_features[i],
                    images_hw[ref_index],
                    target_boxes2d[ref_index],
                    target_classes[ref_index],
                )
                for i, ref_index in enumerate(ref_indices)
            ]

        key_proposals = key_detector_out.proposals.boxes
        ref_proposals = [ref.proposals.boxes for ref in ref_detector_out]
        key_target_boxes = target_boxes2d[key_index]
        ref_target_boxes = [
            target_boxes2d[ref_index] for ref_index in ref_indices
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
        assert (
            ref_embeddings is not None
            and key_track_ids is not None
            and ref_track_ids is not None
        )

        predictions, targets, labels = self.bbox_3d_head(
            features=key_features,
            det_boxes=key_proposals,
            intrinsics=intrinsics[key_index],
            target_boxes=key_target_boxes,
            target_boxes3d=target_boxes3d[key_index],
            target_class_ids=target_classes[key_index],
        )
        detector_3d_out = torch.cat(predictions)
        assert targets is not None and labels is not None

        return FasterRCNNCC3DTOut(
            detector_3d_out=detector_3d_out,
            detector_3d_target=targets,
            detector_3d_labels=labels,
            qdtrack_out=FasterRCNNQDTrackOut(
                detector_out=key_detector_out,
                key_images_hw=images_hw[key_index],
                key_target_boxes=key_target_boxes,
                key_embeddings=key_embeddings,
                ref_embeddings=ref_embeddings,
                key_track_ids=key_track_ids,
                ref_track_ids=ref_track_ids,
            ),
        )

    def _forward_test(
        self,
        images: Tensor,
        images_hw: list[list[tuple[int, int]]],
        intrinsics: Tensor,
        extrinsics: Tensor,
        frame_ids: list[list[int]],
    ) -> Track3DOut:
        """Forward inference stage."""
        # Curretnly only work with single batch per gpu
        # (N, 1, 3, H, W) -> (N, 3, H, W)
        images = images.squeeze(1)
        # (N, 1, 3, 3) -> (N, 3, 3)
        intrinsics = intrinsics.squeeze(1)
        # (N, 1, 4, 4) -> (N, 4, 4)
        extrinsics = extrinsics.squeeze(1)
        # (N, 1) -> (N,)
        images_hw_list: list[tuple[int, int]] = sum(images_hw, [])
        frame_ids_list: list[int] = sum(frame_ids, [])

        features = self.basemodel(images)
        features = self.fpn(features)
        _, roi, proposals, _, _, _ = self.faster_rcnn_head(
            features, images_hw_list
        )

        boxes_2d, scores_2d, class_ids = self.roi2det(
            *roi, proposals.boxes, images_hw_list
        )

        predictions, _, _ = self.bbox_3d_head(features, det_boxes=boxes_2d)

        boxes_3d, scores_3d = self.roi2det_3d(
            predictions, boxes_2d, class_ids, intrinsics
        )

        embeddings, _, _, _ = self.qdtrack(features, boxes_2d)

        outs = self.track_graph(
            embeddings,
            boxes_2d,
            scores_2d,
            boxes_3d,
            scores_3d,
            class_ids,
            frame_ids_list,
            extrinsics,
        )
        return outs

    def __call__(
        self,
        images: list[Tensor] | Tensor,
        images_hw: list[list[tuple[int, int]]],
        intrinsics: list[Tensor] | Tensor,
        extrinsics: Tensor | None = None,
        frame_ids: list[list[int]] | None = None,
        boxes2d: list[list[Tensor]] | None = None,
        boxes3d: list[list[Tensor]] | None = None,
        boxes3d_classes: list[list[Tensor]] | None = None,
        boxes3d_track_ids: list[list[Tensor]] | None = None,
        keyframes: None | list[list[bool]] | None = None,
    ) -> FasterRCNNCC3DTOut | Track3DOut:
        """Type definition for call implementation."""
        return self._call_impl(
            images,
            images_hw,
            intrinsics,
            extrinsics,
            frame_ids,
            boxes2d,
            boxes3d,
            boxes3d_classes,
            boxes3d_track_ids,
            keyframes,
        )
