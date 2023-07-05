"""CC-3DT graph."""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from vis4d.op.box.box2d import bbox_iou
from vis4d.op.detect3d.filter import bev_3d_nms, filter_distance
from vis4d.op.geometry.rotation import (
    rotate_orientation,
    rotate_velocities,
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from vis4d.op.geometry.transform import transform_points
from vis4d.op.track.assignment import TrackIDCounter, greedy_assign
from vis4d.op.track.matching import calc_bisoftmax_affinity
from vis4d.op.track.motion.kalman_filter import predict
from vis4d.op.track3d.motion.kf3d import (
    kf3d_init,
    kf3d_init_mean_cov,
    kf3d_predict,
    kf3d_update,
)
from vis4d.state.track3d.cc_3dt import CC3DTrackMemory, CC3DTrackState


class Track3DOut(NamedTuple):
    """Output of track 3D model."""

    boxes_3d: Tensor
    velocities: Tensor
    class_ids: Tensor
    scores_3d: Tensor
    track_ids: Tensor


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
    oritentation = matrix_to_quaternion(
        euler_angles_to_matrix(boxes_3d[:, 6:9])
    )

    return Track3DOut(
        boxes_3d=torch.cat([center, dims, oritentation], dim=1),
        velocities=boxes_3d[:, 9:12],
        class_ids=class_ids,
        scores_3d=scores_3d,
        track_ids=track_ids,
    )


class CC3DTrackGraph:
    """CC-3DT tracking graph."""

    def __init__(
        self,
        memory_size: int = 10,
        memory_momentum: float = 0.8,
        motion_model: str = "KF3D",
        motion_dims: int = 7,
        num_frames: int = 5,
        pure_det: bool = False,
    ) -> None:
        """Creates an instance of the class."""
        assert 0 <= memory_momentum <= 1.0
        self.memory_size = memory_size
        self.memory_momentum = memory_momentum
        self.track = CC3DTrackAssociation()
        self.track_memory = CC3DTrackMemory(
            memory_limit=memory_size,
            motion_dims=motion_dims,
            num_frames=num_frames,
        )
        self.motion_model = motion_model
        self.motion_dims = motion_dims
        self.num_frames = num_frames
        self.pure_det = pure_det

        if self.motion_model == "KF3D":
            (
                self._motion_mat,  # F
                self._update_mat,  # H
                self._cov_motion_q,  # Q
                self._cov_project_r,  # R
            ) = kf3d_init(self.motion_dims)
        else:
            # TODO: add VeloLSTM
            raise NotImplementedError

    def _update_memory(
        self,
        frame_id: int,
        track_id: int,
        update_attr: str,
        update_value: Tensor,
    ) -> None:
        """Update track memory."""
        track_indice = (
            self.track_memory.frames[frame_id].track_ids == track_id
        ).nonzero(as_tuple=False)[-1]
        frame = self.track_memory.frames[frame_id]
        state_value = list(getattr(frame, update_attr))
        state_value[track_indice] = update_value
        self.track_memory.replace_frame(
            frame_id, update_attr, torch.stack(state_value)
        )

    def _update_track(
        self,
        frame_id: int,
        track_ids: Tensor,
        match_ids: Tensor,
        boxes_2d: Tensor,
        camera_ids: Tensor,
        scores_2d: Tensor,
        boxes_3d: Tensor,
        scores_3d: Tensor,
        class_ids: Tensor,
        embeddings: Tensor,
        obs_boxes_3d: Tensor,
        fps: int,
    ) -> CC3DTrackState:
        """Update track."""
        motion_states_list = []
        motion_hidden_list = []
        vel_histories_list = []
        velocities_list = []
        last_frames_list = []
        acc_frames_list = []
        for i, track_id in enumerate(track_ids):
            bbox_3d = boxes_3d[i]
            obs_3d = obs_boxes_3d[i]
            if track_id in match_ids:
                # update track
                tracks, _ = self.track_memory.get_track(track_id)
                track = tracks[-1]

                mean, covariance = kf3d_update(
                    self._update_mat.to(obs_3d.device),
                    self._cov_project_r.to(obs_3d.device),
                    track.motion_states[0],
                    track.motion_hidden[0],
                    obs_3d,
                )

                pd_box_3d = mean[: self.motion_dims]

                boxes_3d[i][:6] = pd_box_3d[:6]
                boxes_3d[i][8] = pd_box_3d[6]

                pred_loc, _ = predict(
                    self._motion_mat.to(obs_3d.device),
                    self._cov_motion_q.to(obs_3d.device),
                    mean,
                    covariance,
                )
                boxes_3d[i][9:12] = (pred_loc[:3] - mean[:3]) * fps
                prev_obs = torch.cat(
                    [track.boxes_3d[0, :6], track.boxes_3d[0, 8].unsqueeze(0)]
                )
                velocity = (pd_box_3d - prev_obs) / (
                    frame_id - track.last_frames[0]
                )
                velocities_list.append(
                    (track.velocities[0] * track.acc_frames[0] + velocity)
                    / (track.acc_frames[0] + 1)
                )
                acc_frames_list.append(track.acc_frames[0] + 1)

                embeddings[i] = (
                    1 - self.memory_momentum
                ) * track.embeddings + self.memory_momentum * embeddings[i]

                motion_states_list.append(mean)
                motion_hidden_list.append(covariance)
                vel_histories_list.append(
                    torch.zeros(self.num_frames, self.motion_dims).to(
                        obs_3d.device
                    )
                )
            else:
                # create track
                if self.motion_model == "KF3D":
                    mean, covariance = kf3d_init_mean_cov(
                        obs_3d, self.motion_dims
                    )
                    motion_states_list.append(mean)
                    motion_hidden_list.append(covariance)
                else:
                    raise NotImplementedError
                vel_histories_list.append(
                    torch.zeros(self.num_frames, self.motion_dims).to(
                        obs_3d.device
                    )
                )
                velocities_list.append(
                    torch.zeros(self.motion_dims, device=bbox_3d.device)
                )
                acc_frames_list.append(torch.zeros(1, device=bbox_3d.device))
            last_frames_list.append(frame_id)

        motion_states = torch.stack(motion_states_list)
        motion_hidden = torch.stack(motion_hidden_list)
        velocities = torch.stack(velocities_list)
        vel_histories = torch.stack(vel_histories_list)
        last_frames = torch.tensor(last_frames_list, device=boxes_2d.device)
        acc_frames = torch.tensor(acc_frames_list, device=boxes_2d.device)

        return CC3DTrackState(
            track_ids,
            boxes_2d,
            camera_ids,
            scores_2d,
            boxes_3d,
            scores_3d,
            class_ids,
            embeddings,
            motion_states,
            motion_hidden,
            vel_histories,
            velocities,
            last_frames,
            acc_frames,
        )

    def _motion_predict(
        self,
        cur_memory: CC3DTrackState,
        index: int,
        track_id: int,
        device: torch.device,
        update: bool = True,
    ) -> Tensor:
        """Motion prediction."""
        if self.motion_model == "KF3D":
            pd_box_3d, cov = kf3d_predict(
                self._motion_mat.to(device),
                self._cov_motion_q.to(device),
                cur_memory.motion_states[index],
                cur_memory.motion_hidden[index],
            )
            if update:
                _, fids = self.track_memory.get_track(track_id)

                if len(fids) > 0:
                    self._update_memory(
                        fids[-1], track_id, "motion_states", pd_box_3d
                    )
                    self._update_memory(
                        fids[-1], track_id, "motion_hidden", cov
                    )
        else:
            raise NotImplementedError

        return pd_box_3d

    def __call__(
        self,
        embeddings_list: list[Tensor],
        boxes_2d_list: list[Tensor],
        scores_2d_list: list[Tensor],
        boxes_3d_list: list[Tensor],
        scores_3d_list: list[Tensor],
        class_ids_list: list[Tensor],
        frame_ids: list[int],
        extrinsics: Tensor,
        class_range_map: None | Tensor = None,
        fps: int = 2,
    ) -> Track3DOut:
        """Forward function during testing."""
        (
            boxes_2d,
            camera_ids,
            scores_2d,
            boxes_3d,
            scores_3d,
            class_ids,
            embeddings,
        ) = cam_to_global(
            boxes_2d_list,
            scores_2d_list,
            boxes_3d_list,
            scores_3d_list,
            class_ids_list,
            embeddings_list,
            extrinsics,
            class_range_map,
        )

        if self.pure_det:
            return get_track_3d_out(
                boxes_3d,
                class_ids,
                scores_2d * scores_3d,
                torch.zeros_like(class_ids),
            )

        # merge multi-view boxes
        keep_indices = bev_3d_nms(
            boxes_3d,
            scores_2d * scores_3d,
            class_ids,
        )

        boxes_2d = boxes_2d[keep_indices]
        camera_ids = camera_ids[keep_indices]
        scores_2d = scores_2d[keep_indices]
        boxes_3d = boxes_3d[keep_indices]
        scores_3d = scores_3d[keep_indices]
        class_ids = class_ids[keep_indices]
        embeddings = embeddings[keep_indices]

        for frame_id in frame_ids:
            assert (
                frame_id == frame_ids[0]
            ), "All cameras should have same frame_id."
        frame_id = frame_ids[0]

        # reset graph at begin of sequence
        if frame_id == 0:
            self.track_memory.reset()
            TrackIDCounter.reset()

        cur_memory = self.track_memory.get_current_tracks(boxes_2d.device)

        memory_boxes_3d = torch.cat(
            [
                cur_memory.boxes_3d[:, :6],
                cur_memory.boxes_3d[:, 8].unsqueeze(1),
            ],
            dim=1,
        )

        if len(cur_memory.track_ids) > 0:
            memory_boxes_3d_predict = memory_boxes_3d.clone()
            for i, track_id in enumerate(cur_memory.track_ids):
                pd_box_3d = self._motion_predict(
                    cur_memory, i, track_id, boxes_2d.device
                )
                memory_boxes_3d_predict[i, :3] += pd_box_3d[self.motion_dims :]
        else:
            memory_boxes_3d_predict = torch.empty(
                (0, 7), device=boxes_2d.device
            )

        obs_boxes_3d = torch.cat(
            [boxes_3d[:, :6], boxes_3d[:, 8].unsqueeze(1)], dim=1
        )

        track_ids, match_ids, filter_indices = self.track(
            boxes_2d,
            camera_ids,
            scores_2d,
            obs_boxes_3d,
            scores_3d,
            class_ids,
            embeddings,
            memory_boxes_3d,
            cur_memory.track_ids,
            cur_memory.class_ids,
            cur_memory.embeddings,
            memory_boxes_3d_predict,
            cur_memory.velocities,
        )

        data = self._update_track(
            frame_id,
            track_ids,
            match_ids,
            boxes_2d[filter_indices],
            camera_ids[filter_indices],
            scores_2d[filter_indices],
            boxes_3d[filter_indices],
            scores_3d[filter_indices],
            class_ids[filter_indices],
            embeddings[filter_indices],
            obs_boxes_3d[filter_indices],
            fps,
        )

        self.track_memory.update(data)

        tracks = self.track_memory.frames[-1]

        # handle vanished tracklets
        cur_memory = self.track_memory.get_current_tracks(
            device=track_ids.device
        )
        for i, track_id in enumerate(cur_memory.track_ids):
            if frame_id > cur_memory.last_frames[i] and track_id > -1:
                pd_box_3d = self._motion_predict(
                    cur_memory, i, track_id, boxes_2d.device
                )

                _, fids = self.track_memory.get_track(track_id)

                new_box_3d = list(cur_memory.boxes_3d)[i]
                new_box_3d[:6] = pd_box_3d[:6]
                new_box_3d[8] = pd_box_3d[6]
                self._update_memory(fids[-1], track_id, "boxes_3d", new_box_3d)

        # update 3D score
        track_scores_3d = tracks.scores_3d * tracks.scores

        return get_track_3d_out(
            tracks.boxes_3d,
            tracks.class_ids,
            track_scores_3d,
            tracks.track_ids,
        )


# @torch.jit.script TODO
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
                F.pairwise_distance(
                    obsv_boxes_3d[:, :3],
                    memory_box_3d_predict[:3],
                    keepdim=True,
                )
            )
        centroid_weight = torch.cat(centroid_weight_list, dim=1)
        centroid_weight = torch.exp(-centroid_weight / 10.0)

        # Moving distance should be aligned
        motion_weight_list = []
        obsv_velocities = (
            obsv_boxes_3d[:, :3, None]
            - memory_boxes_3d[:, :3, None].transpose(2, 0)
        ).transpose(1, 2)
        for v in obsv_velocities:
            motion_weight_list.append(
                F.pairwise_distance(v, memory_velocities[:, :3]).unsqueeze(0)
            )
        motion_weight = torch.cat(motion_weight_list, dim=0)
        motion_weight = torch.exp(-motion_weight / 5.0)

        # Moving direction should be aligned
        # Set to 0.5 when two vector not within +-90 degree
        cos_sim_list = []
        obsv_direct = (
            obsv_boxes_3d[:, :2, None]
            - memory_boxes_3d[:, :2, None].transpose(2, 0)
        ).transpose(1, 2)
        for d in obsv_direct:
            cos_sim_list.append(
                F.cosine_similarity(d, memory_velocities[:, :2]).unsqueeze(0)
            )
        cos_sim = torch.cat(cos_sim_list, dim=0)
        cos_sim += 1.0
        cos_sim /= 2.0

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
        memory_boxes_3d: Tensor,
        memory_track_ids: Tensor,
        memory_class_ids: Tensor,
        memory_embeddings: Tensor,
        memory_boxes_3d_predict: Tensor,
        memory_velocities: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Process inputs, match detections with existing tracks.

        Args:
            detections (Tensor): [N, 4] detected boxes.
            detection_scores (Tensor): [N,] confidence scores.
            detection_class_ids (Tensor): [N,] class indices.
            detection_embeddings (Tensor): [N, C] appearance embeddings.
            memory_track_ids (Tensor): [M,] track ids in memory.
            memory_class_ids (Tensor): [M,] class indices in memory.
            memory_embeddings (Tensor): [M, C] appearance embeddings in
                memory.

        Returns:
            tuple[Tensor, Tensor]: track ids of active tracks,
                selected detection indices corresponding to tracks.
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

        if len(detections) == 0:
            return (
                torch.empty((0,), dtype=torch.long, device=detections.device),
                torch.empty((0,), dtype=torch.long, device=detections.device),
                torch.empty((0,), dtype=torch.long, device=detections.device),
            )

        # match if buffer is not empty
        if len(memory_track_ids) > 0:
            # Box 3D
            bbox3d_weight_list = []
            for memory_box_3d_predict in memory_boxes_3d_predict:
                bbox3d_weight_list.append(
                    F.pairwise_distance(
                        detections_3d,
                        memory_box_3d_predict,
                        keepdim=True,
                    )
                )
            bbox3d_weight = torch.cat(bbox3d_weight_list, dim=1)
            scores_iou = torch.exp(-bbox3d_weight / 10.0)

            # Depth Ordering
            scores_depth = self.depth_ordering(
                detections_3d,
                memory_boxes_3d,
                memory_boxes_3d_predict,
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
            affinity_scores *= (scores_iou > 0.0).float()
            affinity_scores *= (scores_depth > 0.0).float()
            if self.with_cats:
                affinity_scores *= scores_cats

            ids = greedy_assign(
                detection_scores * detection_scores_3d,
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
        match_ids = ids[ids > -1]
        new_inds = (ids == -1) & (detection_scores > self.init_score_thr)
        ids[new_inds] = TrackIDCounter.get_ids(
            new_inds.sum(), device=ids.device  # type: ignore
        )
        return ids, match_ids, permute_inds


def cam_to_global(
    boxes_2d_list: list[Tensor],
    scores_2d_list: list[Tensor],
    boxes_3d_list: list[Tensor],
    scores_3d_list: list[Tensor],
    class_ids_list: list[Tensor],
    embeddings_list: list[Tensor],
    extrinsics: Tensor,
    class_range_map: None | Tensor = None,
) -> tuple[Tensor, ...]:
    """Convert camera coordinates to global coordinates."""
    camera_ids_list = []
    if sum(len(b) for b in boxes_3d_list) != 0:
        for i, boxes_3d in enumerate(boxes_3d_list):
            if len(boxes_3d) != 0:
                # filter out boxes that are too far away
                if class_range_map is not None:
                    valid_boxes = filter_distance(
                        class_ids_list[i], boxes_3d, class_range_map
                    )
                    boxes_2d_list[i] = boxes_2d_list[i][valid_boxes]
                    scores_2d_list[i] = scores_2d_list[i][valid_boxes]
                    boxes_3d_list[i] = boxes_3d[valid_boxes]
                    scores_3d_list[i] = scores_3d_list[i][valid_boxes]
                    class_ids_list[i] = class_ids_list[i][valid_boxes]
                    embeddings_list[i] = embeddings_list[i][valid_boxes]

                # move 3D boxes to world coordinates
                boxes_3d_list[i][:, :3] = transform_points(
                    boxes_3d_list[i][:, :3], extrinsics[i]
                )
                boxes_3d_list[i][:, 6:9] = rotate_orientation(
                    boxes_3d_list[i][:, 6:9], extrinsics[i]
                )
                boxes_3d_list[i][:, 9:12] = rotate_velocities(
                    boxes_3d_list[i][:, 9:12], extrinsics[i]
                )

                # add camera id
                camera_ids_list.append(
                    (torch.ones(len(boxes_2d_list[i])) * i).to(
                        boxes_2d_list[i].device
                    )
                )

    boxes_2d = torch.cat(boxes_2d_list)
    camera_ids = torch.cat(camera_ids_list)
    scores_2d = torch.cat(scores_2d_list)
    boxes_3d = torch.cat(boxes_3d_list)
    scores_3d = torch.cat(scores_3d_list)
    class_ids = torch.cat(class_ids_list)
    embeddings = torch.cat(embeddings_list)
    return (
        boxes_2d,
        camera_ids,
        scores_2d,
        boxes_3d,
        scores_3d,
        class_ids,
        embeddings,
    )
