"""CC-3DT model implementation.

This file composes the operations associated with CC-3DT
`https://arxiv.org/abs/2212.01247` into the full model implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn

from vis4d.data.const import AxisMode
from vis4d.model.track.qdtrack import FasterRCNNQDTrackOut
from vis4d.op.base import BaseModel, ResNet
from vis4d.op.box.anchor import AnchorGenerator
from vis4d.op.box.box2d import bbox_area, bbox_clip
from vis4d.op.box.box3d import boxes3d_to_corners, transform_boxes3d
from vis4d.op.box.encoder import DeltaXYWHBBoxDecoder
from vis4d.op.detect3d.qd_3dt import QD3DTBBox3DHead, RoI2Det3D
from vis4d.op.detect3d.util import bev_3d_nms
from vis4d.op.detect.faster_rcnn import FasterRCNNHead
from vis4d.op.detect.rcnn import RCNNHead, RoI2Det
from vis4d.op.fpp import FPN
from vis4d.op.geometry.projection import project_points
from vis4d.op.geometry.rotation import (
    quaternion_to_matrix,
    rotation_matrix_yaw,
)
from vis4d.op.geometry.transform import inverse_rigid_transform
from vis4d.op.track3d.cc_3dt import (
    CC3DTrackAssociation,
    cam_to_global,
    get_track_3d_out,
)
from vis4d.op.track3d.common import Track3DOut
from vis4d.op.track.qdtrack import QDTrackHead
from vis4d.state.track3d.cc_3dt import CC3DTrackGraph

from ..track.util import split_key_ref_indices


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
        qdtrack_head: QDTrackHead | None = None,
        track_graph: CC3DTrackGraph | None = None,
        pure_det: bool = False,
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
            track_graph (CC3DTrackGraph, optional): Track graph. Defaults to
                None. If None, will use default CC3DTrackGraph.
            pure_det (bool, optional): Whether to use pure detection. Defaults
                to False.
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

        self.qdtrack_head = (
            QDTrackHead() if qdtrack_head is None else qdtrack_head
        )

        self.track_graph = (
            CC3DTrackGraph() if track_graph is None else track_graph
        )

        self.pure_det = pure_det

    def forward(
        self,
        images: list[Tensor],
        images_hw: list[list[tuple[int, int]]],
        intrinsics: list[Tensor],
        extrinsics: list[Tensor] | None = None,
        frame_ids: list[int] | None = None,
        boxes2d: list[list[Tensor]] | None = None,
        boxes3d: list[list[Tensor]] | None = None,
        boxes3d_classes: list[list[Tensor]] | None = None,
        boxes3d_track_ids: list[list[Tensor]] | None = None,
        keyframes: None | list[list[bool]] | None = None,
    ) -> FasterRCNNCC3DTOut | Track3DOut:
        """Forward."""
        if self.training:
            assert (
                boxes2d is not None
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

        assert extrinsics is not None and frame_ids is not None
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
        images_list: list[Tensor],
        images_hw: list[list[tuple[int, int]]],
        intrinsics_list: list[Tensor],
        extrinsics_list: list[Tensor],
        frame_ids: list[int],
    ) -> Track3DOut:
        """Forward inference stage.

        Curretnly only work with single batch per gpu.
        """
        # (N, 1, 3, H, W) -> (N, 3, H, W)
        images = torch.cat(images_list)
        # (N, 1, 3, 3) -> (N, 3, 3)
        intrinsics = torch.cat(intrinsics_list)
        # (N, 1, 4, 4) -> (N, 4, 4)
        extrinsics = torch.cat(extrinsics_list)
        # (N, 1) -> (N,)
        frame_id = frame_ids[0]
        images_hw_list: list[tuple[int, int]] = sum(images_hw, [])

        features = self.basemodel(images)
        features = self.fpn(features)
        _, roi, proposals, _, _, _ = self.faster_rcnn_head(
            features, images_hw_list
        )

        boxes_2d_list, scores_2d_list, class_ids_list = self.roi2det(
            *roi, proposals.boxes, images_hw_list
        )

        predictions, _, _ = self.bbox_3d_head(
            features, det_boxes=boxes_2d_list
        )

        boxes_3d_list, scores_3d_list = self.roi2det_3d(
            predictions, boxes_2d_list, class_ids_list, intrinsics
        )

        embeddings_list, _, _, _ = self.qdtrack_head(features, boxes_2d_list)

        # Assign camera id
        camera_ids_list = []
        for i, boxes_2d in enumerate(boxes_2d_list):
            camera_ids_list.append(
                (torch.mul(torch.ones(len(boxes_2d)), i)).to(boxes_2d.device)
            )

        # Move 3D boxes to world coordinate
        boxes_3d_list = cam_to_global(boxes_3d_list, extrinsics)

        # Merge boxes from all cameras
        boxes_2d = torch.cat(boxes_2d_list)
        scores_2d = torch.cat(scores_2d_list)
        camera_ids = torch.cat(camera_ids_list)
        boxes_3d = torch.cat(boxes_3d_list)
        scores_3d = torch.cat(scores_3d_list)
        class_ids = torch.cat(class_ids_list)
        embeddings = torch.cat(embeddings_list)

        if self.pure_det:
            return get_track_3d_out(
                boxes_3d, class_ids, scores_3d, torch.zeros_like(class_ids)
            )

        # 3D NMS in world coordinate
        keep_indices = bev_3d_nms(
            center_x=boxes_3d[:, 0].unsqueeze(1),
            center_y=boxes_3d[:, 1].unsqueeze(1),
            width=boxes_3d[:, 4].unsqueeze(1),
            length=boxes_3d[:, 5].unsqueeze(1),
            angle=180.0 / torch.pi * boxes_3d[:, 8].unsqueeze(1),
            scores=scores_2d * scores_3d,
        )

        boxes_2d = boxes_2d[keep_indices]
        scores_2d = scores_2d[keep_indices]
        camera_ids = camera_ids[keep_indices]
        boxes_3d = boxes_3d[keep_indices]
        scores_3d = scores_3d[keep_indices]
        class_ids = class_ids[keep_indices]
        embeddings = embeddings[keep_indices]

        outs = self.track_graph(
            boxes_2d,
            scores_2d,
            camera_ids,
            boxes_3d,
            scores_3d,
            class_ids,
            embeddings,
            frame_id,
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


class CC3DT(nn.Module):
    """CC-3DT with custom detection results."""

    def __init__(
        self,
        basemodel: BaseModel | None = None,
        qdtrack_head: QDTrackHead | None = None,
        track_graph: CC3DTrackGraph | None = None,
        detection_range: Sequence[float] | None = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            basemodel (BaseModel, optional): Base model network. Defaults to
                None. If None, will use ResNet50.
            qdtrack_head (QDTrack, optional): QDTrack head. Defaults to None.
                If None, will use default QDTrackHead.
            track_graph (CC3DTrackGraph, optional): Track graph. Defaults to
                None. If None, will use default CC3DTrackGraph.
            detection_range (Sequence[float], optional): Detection range for
                each class. Defaults to None.
        """
        super().__init__()
        self.basemodel = (
            ResNet(resnet_name="resnet50", pretrained=True, trainable_layers=3)
            if basemodel is None
            else basemodel
        )

        self.fpn = FPN(self.basemodel.out_channels[2:], 256)

        self.qdtrack_head = (
            QDTrackHead() if qdtrack_head is None else qdtrack_head
        )

        self.track_graph = track_graph or CC3DTrackGraph(
            track=CC3DTrackAssociation(init_score_thr=0.2, obj_score_thr=0.1),
            update_3d_score=False,
            add_backdrops=False,
        )

        self.detection_range = detection_range

    def forward(
        self,
        images_list: list[Tensor],
        images_hw: list[list[tuple[int, int]]],
        intrinsics_list: list[Tensor],
        extrinsics_list: list[Tensor],
        frame_ids: list[int],
        pred_boxes3d: list[list[Tensor]],
        pred_boxes3d_classes: list[list[Tensor]],
        pred_boxes3d_scores: list[list[Tensor]],
        pred_boxes3d_velocities: list[list[Tensor]],
    ) -> Track3DOut:
        """Forward inference stage.

        Curretnly only work with single batch per gpu.
        """
        # (N, 1, 3, H, W) -> (N, 3, H, W)
        images = torch.cat(images_list)
        # (N, 1, 3, 3) -> (N, 3, 3)
        intrinsics = torch.cat(intrinsics_list)
        # (N, 1, 4, 4) -> (N, 4, 4)
        extrinsics = torch.cat(extrinsics_list)
        # (N, 1) -> (N,)
        frame_id = frame_ids[0]
        images_hw_list: list[tuple[int, int]] = sum(images_hw, [])

        features = self.basemodel(images)
        features = self.fpn(features)

        # (1, 1, B,) -> (B,)
        boxes_3d = pred_boxes3d[0][0]
        class_ids = pred_boxes3d_classes[0][0]
        scores_3d = pred_boxes3d_scores[0][0]
        velocities = pred_boxes3d_velocities[0][0]

        # Get 2D boxes and assign camera id
        global_to_cams = inverse_rigid_transform(extrinsics)

        boxes_3d_list = []
        boxes_2d_list = []
        class_ids_list = []
        scores_list = []
        camera_ids_list = []
        for i, global_to_cam in enumerate(global_to_cams):
            boxes3d_cam = transform_boxes3d(
                boxes_3d,
                global_to_cam,
                source_axis_mode=AxisMode.ROS,
                target_axis_mode=AxisMode.OPENCV,
            )

            corners = boxes3d_to_corners(
                boxes3d_cam, axis_mode=AxisMode.OPENCV
            )

            corners_2d = project_points(corners, intrinsics[i])

            boxes_2d = self._to_boxes2d(corners_2d)
            boxes_2d = bbox_clip(boxes_2d, images_hw_list[i], 1)

            mask = (
                (boxes3d_cam[:, 2] > 0)
                & (bbox_area(boxes_2d) > 0)
                & (
                    bbox_area(boxes_2d)
                    < (images_hw_list[i][0] - 1) * (images_hw_list[i][1] - 1)
                )
                & self._filter_distance(class_ids, boxes3d_cam)
            )

            cc_3dt_boxes_3d = boxes_3d.new_zeros(len(boxes_2d[mask]), 12)
            cc_3dt_boxes_3d[:, :3] = boxes_3d[mask][:, :3]
            # WLH -> HWL
            cc_3dt_boxes_3d[:, 3:6] = boxes_3d[mask][:, [5, 3, 4]]
            cc_3dt_boxes_3d[:, 6:9] = rotation_matrix_yaw(
                quaternion_to_matrix(boxes_3d[mask][:, 6:]), AxisMode.ROS
            )
            cc_3dt_boxes_3d[:, 9:] = velocities[mask]

            boxes_3d_list.append(cc_3dt_boxes_3d)
            boxes_2d_list.append(boxes_2d[mask])
            class_ids_list.append(class_ids[mask])
            scores_list.append(scores_3d[mask])
            camera_ids_list.append(
                (torch.mul(torch.ones(len(cc_3dt_boxes_3d)), i)).to(
                    boxes_2d.device
                )
            )

        embeddings_list, _, _, _ = self.qdtrack_head(features, boxes_2d_list)

        boxes_3d = torch.cat(boxes_3d_list)
        boxes_2d = torch.cat(boxes_2d_list)
        camera_ids = torch.cat(camera_ids_list)
        scores = torch.cat(scores_list)
        class_ids = torch.cat(class_ids_list)
        embeddings = torch.cat(embeddings_list)

        # Select project boxes2d according to bbox area
        keep_indices = embeddings.new_ones(len(boxes_3d)).bool()
        boxes_2d_area = bbox_area(boxes_2d)
        for i, box3d in enumerate(boxes_3d):
            for same_idx in (
                (box3d[:3] == boxes_3d[:, :3]).all(dim=1).nonzero()
            ):
                if (
                    same_idx != i
                    and boxes_2d_area[same_idx] > boxes_2d_area[i]
                ):
                    keep_indices[i] = False
                    break

        boxes_3d = boxes_3d[keep_indices]
        boxes_2d = boxes_2d[keep_indices]
        camera_ids = camera_ids[keep_indices]
        scores = scores[keep_indices]
        class_ids = class_ids[keep_indices]
        embeddings = embeddings[keep_indices]

        outs = self.track_graph(
            boxes_2d,
            scores,
            camera_ids,
            boxes_3d,
            scores,
            class_ids,
            embeddings,
            frame_id,
        )

        return outs

    def _to_boxes2d(self, corners_2d: Tensor) -> Tensor:
        """Project 3D boxes (Camera coordinates) to 2D boxes."""
        min_x = torch.min(corners_2d[:, :, 0], 1).values.unsqueeze(-1)
        min_y = torch.min(corners_2d[:, :, 1], 1).values.unsqueeze(-1)
        max_x = torch.max(corners_2d[:, :, 0], 1).values.unsqueeze(-1)
        max_y = torch.max(corners_2d[:, :, 1], 1).values.unsqueeze(-1)

        return torch.cat([min_x, min_y, max_x, max_y], dim=1)

    def _filter_distance(
        self, class_ids: Tensor, boxes3d: Tensor, tolerance: float = 2.0
    ) -> Tensor:
        """Filter boxes3d on distance."""
        if self.detection_range is None:
            return torch.ones_like(class_ids, dtype=torch.bool)

        return torch.linalg.norm(  # pylint: disable=not-callable
            boxes3d[:, [0, 2]], dim=1
        ) <= torch.tensor(
            [
                self.detection_range[class_id] + tolerance
                for class_id in class_ids
            ]
        ).to(
            class_ids.device
        )

    def __call__(
        self,
        images_list: list[Tensor],
        images_hw: list[list[tuple[int, int]]],
        intrinsics_list: list[Tensor],
        extrinsics_list: list[Tensor],
        frame_ids: list[int],
        pred_boxes3d: list[list[Tensor]],
        pred_boxes3d_classes: list[list[Tensor]],
        pred_boxes3d_scores: list[list[Tensor]],
        pred_boxes3d_velocities: list[list[Tensor]],
    ) -> Track3DOut:
        """Type definition for call implementation."""
        return self._call_impl(
            images_list,
            images_hw,
            intrinsics_list,
            extrinsics_list,
            frame_ids,
            pred_boxes3d,
            pred_boxes3d_classes,
            pred_boxes3d_scores,
            pred_boxes3d_velocities,
        )
