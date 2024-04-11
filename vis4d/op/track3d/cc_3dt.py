"""CC-3DT graph."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from vis4d.op.box.box2d import bbox_iou
from vis4d.op.geometry.rotation import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    rotate_orientation,
    rotate_velocities,
)
from vis4d.op.geometry.transform import transform_points
from vis4d.op.track.assignment import TrackIDCounter, greedy_assign
from vis4d.op.track.matching import calc_bisoftmax_affinity

from .common import Track3DOut


def get_track_3d_out(
    boxes_3d: Tensor, class_ids: Tensor, scores_3d: Tensor, track_ids: Tensor
) -> Track3DOut:
    """Get track 3D output.

    Args:
        boxes_3d (Tensor): (N, 12): x,y,z,h,w,l,rx,ry,rz,vx,vy,vz
        class_ids (Tensor): (N,)
        scores_3d (Tensor): (N,)
        track_ids (Tensor): (N,)

    Returns:
        Track3DOut: output
    """
    center = boxes_3d[:, :3]
    # HWL -> WLH
    dims = boxes_3d[:, [4, 5, 3]]
    orientation = matrix_to_quaternion(
        euler_angles_to_matrix(boxes_3d[:, 6:9])
    )

    return Track3DOut(
        boxes_3d=[torch.cat([center, dims, orientation], dim=1)],
        velocities=[boxes_3d[:, 9:12]],
        class_ids=[class_ids],
        scores_3d=[scores_3d],
        track_ids=[track_ids],
    )


class CC3DTrackAssociation:
    """Data association relying on quasi-dense instance similarity and 3D clue.

    This class assigns detection candidates to a given memory of existing
    tracks and backdrops.
    Backdrops are low-score detections kept in case they have high
    similarity with a high-score detection in succeeding frames.
    """

    def __init__(
        self,
        init_score_thr: float = 0.8,
        obj_score_thr: float = 0.5,
        match_score_thr: float = 0.5,
        nms_backdrop_iou_thr: float = 0.3,
        nms_class_iou_thr: float = 0.7,
        nms_conf_thr: float = 0.5,
        with_cats: bool = True,
        bbox_affinity_weight: float = 0.5,
    ) -> None:
        """Creates an instance of the class.

        Args:
            init_score_thr (float): Confidence threshold for initializing a new
                track.
            obj_score_thr (float): Confidence treshold s.t. a detection is
                considered in the track / det matching process.
            match_score_thr (float): Similarity score threshold for matching a
                detection to an existing track.
            nms_backdrop_iou_thr (float): Maximum IoU of a backdrop with
                another detection.
            nms_class_iou_thr (float): Maximum IoU of a high score detection
                with another of a different class.
            with_cats (bool): If to consider category information for
                tracking (i.e. all detections within a track must have
                consistent category labels).
            nms_conf_thr (float): Confidence threshold for NMS.
            bbox_affinity_weight (float): Weight of bbox affinity in the
                overall affinity score.
        """
        super().__init__()
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.nms_conf_thr = nms_conf_thr
        self.with_cats = with_cats
        self.bbox_affinity_weight = bbox_affinity_weight
        self.feat_affinity_weight = 1 - bbox_affinity_weight

    def _filter_detections(
        self,
        detections: Tensor,
        camera_ids: Tensor,
        scores: Tensor,
        detections_3d: Tensor,
        scores_3d: Tensor,
        class_ids: Tensor,
        embeddings: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Remove overlapping objects across classes via nms.

        Args:
            detections (Tensor): [N, 4] Tensor of boxes.
            camera_ids (Tensor): [N,] Tensor of camera ids.
            scores (Tensor): [N,] Tensor of confidence scores.
            detections_3d (Tensor): [N, 7] Tensor of 3D boxes.
            scores_3d (Tensor): [N,] Tensor of 3D confidence scores.
            class_ids (Tensor): [N,] Tensor of class ids.
            embeddings (Tensor): [N, C] tensor of appearance embeddings.

        Returns:
            tuple[Tensor]: filtered detections, scores, class_ids,
                embeddings, and filtered indices.
        """
        scores, inds = scores.sort(descending=True)
        (
            detections,
            camera_ids,
            embeddings,
            class_ids,
            detections_3d,
            scores_3d,
        ) = (
            detections[inds],
            camera_ids[inds],
            embeddings[inds],
            class_ids[inds],
            detections_3d[inds],
            scores_3d[inds],
        )
        valids = embeddings.new_ones((len(detections),), dtype=torch.bool)

        ious = bbox_iou(detections, detections)
        valid_ious = torch.eq(
            camera_ids.unsqueeze(1), camera_ids.unsqueeze(0)
        ).int()
        ious *= valid_ious

        for i in range(1, len(detections)):
            if scores[i] < self.obj_score_thr:
                thr = self.nms_backdrop_iou_thr
            else:
                thr = self.nms_class_iou_thr

            if (ious[i, :i] > thr).any():
                valids[i] = False
        detections = detections[valids]
        scores = scores[valids]
        detections_3d = detections_3d[valids]
        scores_3d = scores_3d[valids]
        class_ids = class_ids[valids]
        embeddings = embeddings[valids]
        return (
            detections,
            scores,
            detections_3d,
            scores_3d,
            class_ids,
            embeddings,
            inds[valids],
        )

    @staticmethod
    def depth_ordering(
        obsv_boxes_3d: Tensor,
        memory_boxes_3d_predict: Tensor,
        memory_boxes_3d: Tensor,
        memory_velocities: Tensor,
    ) -> Tensor:
        """Depth ordering matching."""
        # Centroid
        centroid_weight_list = []
        for memory_box_3d_predict in memory_boxes_3d_predict:
            centroid_weight_list.append(
                F.pairwise_distance(  # pylint: disable=not-callable
                    obsv_boxes_3d[:, :3],
                    memory_box_3d_predict[:3],
                    keepdim=True,
                )
            )
        centroid_weight = torch.cat(centroid_weight_list, dim=1)
        centroid_weight = torch.exp(-torch.div(centroid_weight, 10.0))

        # Moving distance should be aligned
        motion_weight_list = []
        obsv_velocities = (
            obsv_boxes_3d[:, :3, None]
            - memory_boxes_3d[:, :3, None].transpose(2, 0)
        ).transpose(1, 2)
        for v in obsv_velocities:
            motion_weight_list.append(
                F.pairwise_distance(  # pylint: disable=not-callable
                    v, memory_velocities[:, :3]
                ).unsqueeze(0)
            )
        motion_weight = torch.cat(motion_weight_list, dim=0)
        motion_weight = torch.exp(-torch.div(motion_weight, 5.0))

        # Moving direction should be aligned
        # Set to 0.5 when two vector not within +-90 degree
        cos_sim_list = []
        obsv_direct = (
            obsv_boxes_3d[:, :2, None]
            - memory_boxes_3d[:, :2, None].transpose(2, 0)
        ).transpose(1, 2)
        for d in obsv_direct:
            cos_sim_list.append(
                F.cosine_similarity(  # pylint: disable=not-callable
                    d, memory_velocities[:, :2]
                ).unsqueeze(0)
            )
        cos_sim = torch.cat(cos_sim_list, dim=0)
        cos_sim = torch.add(cos_sim, 1.0)
        cos_sim = torch.div(cos_sim, 2.0)

        scores_depth = (
            cos_sim * centroid_weight + (1.0 - cos_sim) * motion_weight
        )

        return scores_depth

    def __call__(
        self,
        detections: Tensor,
        camera_ids: Tensor,
        detection_scores: Tensor,
        detections_3d: Tensor,
        detection_scores_3d: Tensor,
        detection_class_ids: Tensor,
        detection_embeddings: Tensor,
        memory_boxes_3d: Tensor | None = None,
        memory_track_ids: Tensor | None = None,
        memory_class_ids: Tensor | None = None,
        memory_embeddings: Tensor | None = None,
        memory_boxes_3d_predict: Tensor | None = None,
        memory_velocities: Tensor | None = None,
        with_depth_confidence: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Process inputs, match detections with existing tracks.

        Args:
            detections (Tensor): [N, 4] detected boxes.
            camera_ids (Tensor): [N,] camera ids.
            detection_scores (Tensor): [N,] confidence scores.
            detections_3d (Tensor): [N, 7] detected boxes in 3D.
            detection_scores_3d (Tensor): [N,] confidence scores in 3D.
            detection_class_ids (Tensor): [N,] class indices.
            detection_embeddings (Tensor): [N, C] appearance embeddings.
            memory_boxes_3d (Tensor): [M, 7] boxes in memory.
            memory_track_ids (Tensor): [M,] track ids in memory.
            memory_class_ids (Tensor): [M,] class indices in memory.
            memory_embeddings (Tensor): [M, C] appearance embeddings in
                memory.
            memory_boxes_3d_predict (Tensor): [M, 7] predicted boxes in
                memory.
            memory_velocities (Tensor): [M, 7] velocities in memory.

        Returns:
            tuple[Tensor, Tensor]: track ids of active tracks and selected
                detection indices corresponding to tracks.
        """
        (
            detections,
            detection_scores,
            detections_3d,
            detection_scores_3d,
            detection_class_ids,
            detection_embeddings,
            permute_inds,
        ) = self._filter_detections(
            detections,
            camera_ids,
            detection_scores,
            detections_3d,
            detection_scores_3d,
            detection_class_ids,
            detection_embeddings,
        )

        if with_depth_confidence:
            depth_confidence = detection_scores_3d
        else:
            depth_confidence = detection_scores_3d.new_ones(
                len(detection_scores_3d)
            )

        # match if buffer is not empty
        if len(detections) > 0 and memory_boxes_3d is not None:
            assert (
                memory_track_ids is not None
                and memory_class_ids is not None
                and memory_embeddings is not None
                and memory_boxes_3d_predict is not None
                and memory_velocities is not None
            )

            # Box 3D
            bbox3d_weight_list = []
            for memory_box_3d_predict in memory_boxes_3d_predict:
                bbox3d_weight_list.append(
                    F.pairwise_distance(  # pylint: disable=not-callable
                        detections_3d,
                        memory_box_3d_predict,
                        keepdim=True,
                    )
                )
            bbox3d_weight = torch.cat(bbox3d_weight_list, dim=1)
            scores_iou = torch.exp(-torch.div(bbox3d_weight, 10.0))

            # Depth Ordering
            scores_depth = self.depth_ordering(
                detections_3d,
                memory_boxes_3d_predict,
                memory_boxes_3d,
                memory_velocities,
            )

            # match using bisoftmax metric
            similarity_scores = calc_bisoftmax_affinity(
                detection_embeddings,
                memory_embeddings,
                detection_class_ids,
                memory_class_ids,
            )

            if self.with_cats:
                assert (
                    detection_class_ids is not None
                    and memory_class_ids is not None
                ), "Please provide class ids if with_categories=True!"
                cat_same = detection_class_ids.view(
                    -1, 1
                ) == memory_class_ids.view(1, -1)
                scores_cats = cat_same.float()

            affinity_scores = (
                self.bbox_affinity_weight * scores_iou * scores_depth
                + self.feat_affinity_weight * similarity_scores
            )
            affinity_scores /= (
                self.bbox_affinity_weight + self.feat_affinity_weight
            )
            affinity_scores = torch.mul(
                affinity_scores, torch.greater(scores_iou, 0.0).float()
            )
            affinity_scores = torch.mul(
                affinity_scores, torch.greater(scores_depth, 0.0).float()
            )
            if self.with_cats:
                affinity_scores = torch.mul(affinity_scores, scores_cats)

            ids = greedy_assign(
                detection_scores * depth_confidence,
                memory_track_ids,
                affinity_scores,
                self.match_score_thr,
                self.obj_score_thr,
                self.nms_conf_thr,
            )
        else:
            ids = torch.full(
                (len(detections),),
                -1,
                dtype=torch.long,
                device=detections.device,
            )
        new_inds = (ids == -1) & (detection_scores > self.init_score_thr)
        ids[new_inds] = TrackIDCounter.get_ids(
            new_inds.sum(), device=ids.device  # type: ignore
        )
        return ids, permute_inds


def cam_to_global(
    boxes_3d_list: list[Tensor], extrinsics: Tensor
) -> list[Tensor]:
    """Convert camera coordinates to global coordinates."""
    for i, boxes_3d in enumerate(boxes_3d_list):
        if len(boxes_3d) != 0:
            boxes_3d_list[i][:, :3] = transform_points(
                boxes_3d_list[i][:, :3], extrinsics[i]
            )
            boxes_3d_list[i][:, 6:9] = rotate_orientation(
                boxes_3d_list[i][:, 6:9], extrinsics[i]
            )
            boxes_3d_list[i][:, 9:12] = rotate_velocities(
                boxes_3d_list[i][:, 9:12], extrinsics[i]
            )
    return boxes_3d_list
