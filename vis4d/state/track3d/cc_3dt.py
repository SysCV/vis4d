"""Memory for CC-3DT inference."""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from vis4d.op.box.box2d import bbox_iou
from vis4d.op.detect3d.util import bev_3d_nms
from vis4d.op.geometry.rotation import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from vis4d.op.track.motion.kalman_filter import predict
from vis4d.op.track.assignment import TrackIDCounter
from vis4d.op.track3d.common import Track3DOut
from vis4d.op.track3d.motion.kf3d import (
    kf3d_init,
    kf3d_init_mean_cov,
    kf3d_predict,
    kf3d_update,
)
from vis4d.op.track3d.cc_3dt import CC3DTrackAssociation, cam_to_global

from ..track.util import (
    concat_states,
    get_last_tracks,
    merge_tracks,
    update_frames,
)

# TODO: Add VeloLSTM motion state
# class KF3DMotionState(NamedTuple):
#     """KF3D motion state."""

#     mean: Tensor
#     covariance: Tensor

# class VeloLSTMState(NamedTuple):
#     """VeloLSTM motion state."""

#     history: Tensor
#     ref_history: Tensor
#     hidden_pred: Tensor
#     hidden_ref: Tensor


class CC3DTrackState(NamedTuple):
    """CC-3DT Track state."""

    track_ids: Tensor
    boxes: Tensor
    camera_ids: Tensor
    scores: Tensor
    boxes_3d: Tensor
    scores_3d: Tensor
    class_ids: Tensor
    embeddings: Tensor
    motion_states: Tensor
    motion_hidden: Tensor
    vel_histories: Tensor
    velocities: Tensor
    last_frames: Tensor
    acc_frames: Tensor


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
        detection_range: None | list[float] = None,
        fps: int = 2,
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
        self.detection_range = detection_range
        self.fps = fps

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
            self.detection_range,
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
            self.fps,
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


class CC3DTrackMemory:
    """CC-3DT Track Memory."""

    def __init__(
        self,
        memory_limit: int = -1,
        backdrop_memory_limit: int = 1,
        nms_backdrop_iou_thr: float = 0.3,
        motion_dims: int = 7,
        num_frames: int = 5,
    ):
        """Creates an instance of the class."""
        assert memory_limit >= -1
        self.memory_limit = memory_limit
        self.frames: list[CC3DTrackState] = []
        self.backdrop_frames: list[CC3DTrackState] = []
        assert backdrop_memory_limit >= 0
        self.backdrop_memory_limit = backdrop_memory_limit
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr

        self.motion_dims = motion_dims
        self.num_frames = num_frames

    def reset(self) -> None:
        """Empty the memory."""
        self.frames.clear()
        self.backdrop_frames.clear()

    def update(self, data: CC3DTrackState) -> None:
        """Update the track memory with a new state."""
        valid_tracks = torch.nonzero(
            data.track_ids > -1, as_tuple=False
        ).squeeze(1)

        new_tracks = CC3DTrackState(*(entry[valid_tracks] for entry in data))

        self.frames = update_frames(
            self.frames, new_tracks, self.memory_limit  # type: ignore
        )

        # backdrops
        backdrop_tracks = torch.nonzero(
            data.track_ids == -1, as_tuple=False
        ).squeeze(1)

        valid_ious = torch.eq(
            data.camera_ids[backdrop_tracks].unsqueeze(1),
            data.camera_ids.unsqueeze(0),
        ).int()
        ious = bbox_iou(data.boxes[backdrop_tracks], data.boxes)
        ious *= valid_ious

        for i, ind in enumerate(backdrop_tracks):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_tracks[i] = -1
        backdrop_tracks = backdrop_tracks[backdrop_tracks > -1]

        new_backdrops = CC3DTrackState(
            *(entry[backdrop_tracks] for entry in data)
        )
        self.backdrop_frames = update_frames(
            self.backdrop_frames, new_backdrops, self.backdrop_memory_limit  # type: ignore # pylint: disable=line-too-long
        )

    def replace_frame(
        self, frame_id: int, state_attr: str, state_value: Tensor
    ) -> None:
        """Replace the frame of track memory with a new state."""
        self.frames[frame_id] = self.frames[frame_id]._replace(
            **{state_attr: state_value}
        )

    def get_track(
        self, track_id: int
    ) -> tuple[list[CC3DTrackState], list[int]]:
        """Get representation of a single track across memory frames.

        Args:
            track_id (int): track id of query track.

        Returns:
            track (list[CC3DTrackState]): List of track states for given query
            track.
            frame_ids (list[int]): List of frame ids for given query track.
        """
        frame_ids = []
        track = []
        for i, frame in enumerate(self.frames):
            idx = (frame.track_ids == track_id).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                track.append(
                    CC3DTrackState(*(element[idx] for element in frame))
                )
                frame_ids.append(i)
        return track, frame_ids

    def get_empty_frame(
        self, n_tracks: int, device: torch.device
    ) -> CC3DTrackState:
        """Get an empty frame with the correct dimensions."""
        track_ids = torch.empty((n_tracks,), dtype=torch.int64, device=device)
        class_ids = torch.empty((n_tracks,), dtype=torch.int64, device=device)
        camera_ids = torch.empty((n_tracks,), dtype=torch.int64, device=device)
        scores = torch.empty((n_tracks,), device=device)
        boxes = torch.empty((n_tracks, 4), device=device)
        scores_3d = torch.empty((n_tracks,), device=device)
        boxes_3d = torch.empty((n_tracks, 12), device=device)
        embeddings = torch.empty((n_tracks, 1), device=device)
        motion_states = torch.empty(
            (n_tracks, self.motion_dims + 3), device=device
        )
        motion_hidden = torch.empty(
            (n_tracks, self.motion_dims + 3, self.motion_dims + 3),
            device=device,
        )
        vel_histories = torch.empty(
            (n_tracks, self.num_frames, self.motion_dims), device=device
        )
        velocities = torch.empty((n_tracks, self.motion_dims), device=device)
        last_frames = torch.empty((n_tracks,), device=device)
        acc_frames = torch.empty((n_tracks,), device=device)
        return CC3DTrackState(
            track_ids,
            boxes,
            camera_ids,
            scores,
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

    def get_current_tracks(self, device: torch.device) -> CC3DTrackState:
        """Get all active tracks and backdrops in memory."""
        # get last states of all tracks
        if len(self.frames) > 0:
            memory_states = CC3DTrackState(
                *(concat_states(self.frames))  # type: ignore
            )

            last_tracks = CC3DTrackState(*(get_last_tracks(memory_states)))
        else:
            last_tracks = self.get_empty_frame(0, device)

        # add backdrops
        if len(self.backdrop_frames) > 0:
            backdrops = CC3DTrackState(
                *(concat_states(self.backdrop_frames))  # type: ignore
            )

            if backdrops.embeddings.size(1) != last_tracks.embeddings.size(1):
                assert (
                    len(last_tracks.embeddings) == 0
                ), "Unequal shape of backdrop embeddings and track embeddings!"
                last_tracks = last_tracks._replace(
                    embeddings=torch.empty(
                        (0, backdrops.embeddings.size(1)), device=device
                    )
                )

            last_tracks = CC3DTrackState(
                *(merge_tracks(last_tracks, backdrops))
            )
        return last_tracks
