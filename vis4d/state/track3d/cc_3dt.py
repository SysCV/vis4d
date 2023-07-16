"""Memory for CC-3DT inference."""
from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor

from vis4d.common.typing import DictStrAny
from vis4d.model.motion.velo_lstm import VeloLSTM
from vis4d.op.box.box2d import bbox_iou
from vis4d.op.track3d.cc_3dt import CC3DTrackAssociation, get_track_3d_out
from vis4d.op.track3d.common import Track3DOut
from vis4d.op.track.assignment import TrackIDCounter

from .motion import BaseMotionModel, KF3DMotionModel, LSTM3DMotionModel


class CC3DTrackGraph:
    """CC-3DT tracking graph."""

    def __init__(
        self,
        memory_size: int = 10,
        memory_momentum: float = 0.8,
        motion_model: str = "KF3D",
        motion_dims: int = 7,
        num_frames: int = 5,
        fps: int = 2,
    ) -> None:
        """Creates an instance of the class."""
        assert 0 <= memory_momentum <= 1.0
        self.track = CC3DTrackAssociation()
        self.track_memory = CC3DTrackMemory(
            memory_momentum=memory_momentum,
            memory_limit=memory_size,
            motion_model=motion_model,
            motion_dims=motion_dims,
            num_frames=num_frames,
            fps=fps,
        )
        self.motion_dims = motion_dims

    def __call__(
        self,
        boxes_2d: Tensor,
        scores_2d: Tensor,
        camera_ids: Tensor,
        boxes_3d: Tensor,
        scores_3d: Tensor,
        class_ids: Tensor,
        embeddings: Tensor,
        frame_id: int,
    ) -> Track3DOut:
        """Forward function during testing."""
        # reset graph at begin of sequence
        if frame_id == 0:
            self.track_memory.reset()
            TrackIDCounter.reset()

        if not self.track_memory.is_empty():
            (
                _,
                _,
                memo_boxes_3d,
                _,
                memo_class_ids,
                memo_track_ids,
                memo_embeds,
                memo_motion_models,
                memo_velocities,
            ) = self.track_memory.get_tracks(
                boxes_2d.device, add_backdrops=True
            )

            memory_boxes_3d = torch.cat(
                [memo_boxes_3d[:, :6], memo_boxes_3d[:, 8].unsqueeze(1)],
                dim=1,
            )

            memory_track_ids = memo_track_ids
            memory_class_ids = memo_class_ids
            memory_embeddings = memo_embeds

            memory_boxes_3d_predict = memory_boxes_3d.clone()
            for i, memo_motion_model in enumerate(memo_motion_models):
                pd_box_3d = memo_motion_model.predict(
                    update_state=memo_motion_model.age != 0
                )
                memory_boxes_3d_predict[i, :3] += pd_box_3d[self.motion_dims :]

            memory_velocities = memo_velocities

        else:
            memory_boxes_3d = None
            memory_track_ids = None
            memory_class_ids = None
            memory_embeddings = None
            memory_boxes_3d_predict = None
            memory_velocities = None

        obs_boxes_3d = torch.cat(
            [boxes_3d[:, :6], boxes_3d[:, 8].unsqueeze(1)], dim=1
        )

        track_ids, _, filter_indices = self.track(
            boxes_2d,
            camera_ids,
            scores_2d,
            obs_boxes_3d,
            scores_3d,
            class_ids,
            embeddings,
            memory_boxes_3d,
            memory_track_ids,
            memory_class_ids,
            memory_embeddings,
            memory_boxes_3d_predict,
            memory_velocities,
        )

        self.track_memory.update(
            frame_id,
            track_ids,
            boxes_2d[filter_indices],
            scores_2d[filter_indices],
            camera_ids[filter_indices],
            boxes_3d[filter_indices],
            scores_3d[filter_indices],
            class_ids[filter_indices],
            embeddings[filter_indices],
            obs_boxes_3d[filter_indices],
        )

        (
            _,
            scores_2d,
            boxes_3d,
            scores_3d,
            class_ids,
            track_ids,
            _,
            _,
            _,
        ) = self.track_memory.get_tracks(boxes_2d.device, frame_id=frame_id)

        # update 3D score
        track_scores_3d = scores_2d * scores_3d

        return get_track_3d_out(
            boxes_3d, class_ids, track_scores_3d, track_ids
        )


class Track(TypedDict):
    """CC-3DT Track state."""

    box_2d: Tensor
    score_2d: Tensor
    box_3d: Tensor
    score_3d: Tensor
    class_id: Tensor
    embed: Tensor
    motion_model: BaseMotionModel
    velocity: Tensor
    last_frame: int
    acc_frame: int


class CC3DTrackMemory:
    """CC-3DT Track Memory."""

    def __init__(
        self,
        memory_momentum: float = 0.8,
        memory_limit: int = -1,
        backdrop_memory_limit: int = 1,
        nms_backdrop_iou_thr: float = 0.3,
        motion_model: str = "KF3D",
        motion_dims: int = 7,
        num_frames: int = 5,
        fps: int = 2,
    ):
        """Creates an instance of the class."""
        self.memory_momentum = memory_momentum
        assert memory_limit >= -1
        self.memory_limit = memory_limit
        self.tracks: dict[int, Track] = {}
        self.backdrops: list[DictStrAny] = []
        assert backdrop_memory_limit >= 0
        self.backdrop_memory_limit = backdrop_memory_limit
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr

        # Build motion model
        if motion_model == "VeloLSTM":
            lstm_model = VeloLSTM(
                loc_dim=motion_dims,
                weights="./vis4d-workspace/checkpoints_old/velo_lstm_cc_3dt_frcnn_r101_fpn_100e_nusc.pt",
            )
            self.lstm_model = lstm_model

        self.motion_model = motion_model
        self.motion_dims = motion_dims
        self.num_frames = num_frames
        self.fps = fps

    def reset(self) -> None:
        """Empty the memory."""
        self.tracks.clear()
        self.backdrops.clear()

    def is_empty(self) -> bool:
        """Check if the memory is empty."""
        return len(self.tracks) == 0

    def get_tracks(
        self,
        device: torch.device,
        frame_id: int | None = None,
        add_backdrops=False,
    ):
        """Get representation of a single track across memory frames.

        Args:
            track_id (int): track id of query track.

        Returns:
            track (list[CC3DTrackState]): List of track states for given query
            track.
            frame_ids (list[int]): List of frame ids for given query track.
        """
        (
            boxes_2d,
            scores_2d,
            boxes_3d,
            scores_3d,
            class_ids,
            embeds,
            motion_models,
            velocities,
            track_ids,
        ) = ([], [], [], [], [], [], [], [], [])

        for track_id, track in self.tracks.items():
            if frame_id is None or track["last_frame"] == frame_id:
                boxes_2d.append(track["box_2d"].unsqueeze(0))
                scores_2d.append(track["score_2d"].unsqueeze(0))
                boxes_3d.append(track["box_3d"].unsqueeze(0))
                scores_3d.append(track["score_3d"].unsqueeze(0))
                class_ids.append(track["class_id"].unsqueeze(0))
                embeds.append(track["embed"].unsqueeze(0))
                motion_models.append(track["motion_model"])
                velocities.append(track["velocity"].unsqueeze(0))
                track_ids.append(track_id)

        boxes_2d = (
            torch.cat(boxes_2d)
            if len(boxes_2d) > 0
            else torch.empty((0, 4), device=device)
        )
        scores_2d = (
            torch.cat(scores_2d)
            if len(scores_2d) > 0
            else torch.empty((0,), device=device)
        )
        boxes_3d = (
            torch.cat(boxes_3d)
            if len(boxes_3d) > 0
            else torch.empty((0, 12), device=device)
        )
        scores_3d = (
            torch.cat(scores_3d)
            if len(scores_3d) > 0
            else torch.empty((0,), device=device)
        )
        class_ids = (
            torch.cat(class_ids)
            if len(class_ids) > 0
            else torch.empty((0,), device=device)
        )
        embeds = (
            torch.cat(embeds)
            if len(embeds) > 0
            else torch.empty((0,), device=device)
        )
        velocities = (
            torch.cat(velocities)
            if len(velocities) > 0
            else torch.empty((0, self.motion_dims), device=device)
        )
        track_ids = torch.tensor(track_ids, device=device)

        if add_backdrops:
            for backdrop in self.backdrops:
                backdrop_ids = torch.full(
                    (len(backdrop["embeddings"]),),
                    -1,
                    dtype=torch.long,
                    device=device,
                )
                track_ids = torch.cat([track_ids, backdrop_ids])
                boxes_2d = torch.cat([boxes_2d, backdrop["boxes_2d"]])
                scores_2d = torch.cat([scores_2d, backdrop["scores_2d"]])
                boxes_3d = torch.cat([boxes_3d, backdrop["boxes_3d"]])
                scores_3d = torch.cat([scores_3d, backdrop["scores_3d"]])
                class_ids = torch.cat([class_ids, backdrop["class_ids"]])
                embeds = torch.cat([embeds, backdrop["embeddings"]])
                motion_models.extend(backdrop["motion_models"])
                backdrop_vs = torch.zeros_like(
                    backdrop["boxes_3d"][:, : self.motion_dims]
                )
                velocities = torch.cat([velocities, backdrop_vs])

        return (
            boxes_2d,
            scores_2d,
            boxes_3d,
            scores_3d,
            class_ids,
            track_ids,
            embeds,
            motion_models,
            velocities,
        )

    def update(
        self,
        frame_id: int,
        track_ids: Tensor,
        boxes_2d: Tensor,
        scores_2d: Tensor,
        camera_ids: Tensor,
        boxes_3d: Tensor,
        scores_3d: Tensor,
        class_ids: Tensor,
        embeddings: Tensor,
        obs_boxes_3d: Tensor,
    ) -> None:
        """Update the track memory with a new state."""
        valid_tracks = track_ids > -1

        # update memo
        for (
            track_id,
            box_2d,
            score_2d,
            box_3d,
            score_3d,
            class_id,
            embed,
            obs_box_3d,
        ) in zip(
            track_ids[valid_tracks],
            boxes_2d[valid_tracks],
            scores_2d[valid_tracks],
            boxes_3d[valid_tracks],
            scores_3d[valid_tracks],
            class_ids[valid_tracks],
            embeddings[valid_tracks],
            obs_boxes_3d[valid_tracks],
        ):
            track_id = int(track_id)
            if track_id in self.tracks:
                self.update_track(
                    track_id,
                    box_2d,
                    score_2d,
                    box_3d,
                    score_3d,
                    class_id,
                    embed,
                    obs_box_3d,
                    frame_id,
                )
            else:
                self.create_track(
                    track_id,
                    box_2d,
                    score_2d,
                    box_3d,
                    score_3d,
                    class_id,
                    embed,
                    obs_box_3d,
                    frame_id,
                )

        # Handle vanished tracklets
        for track_id, track in self.tracks.items():
            if frame_id > track["last_frame"] and track_id > -1:
                pd_box_3d = track["motion_model"].predict()
                track["box_3d"][:6] = pd_box_3d[:6]
                track["box_3d"][8] = pd_box_3d[6]

        # Backdrops
        backdrop_inds = torch.nonzero(track_ids == -1, as_tuple=False).squeeze(
            1
        )

        valid_ious = torch.eq(
            camera_ids[backdrop_inds].unsqueeze(1),
            camera_ids.unsqueeze(0),
        ).int()
        ious = bbox_iou(boxes_2d[backdrop_inds], boxes_2d)
        ious *= valid_ious

        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]

        backdrop_motion_model = []
        for bd_ind in backdrop_inds:
            backdrop_motion_model.append(
                self.build_motion_model(obs_boxes_3d[bd_ind])
            )

        self.backdrops.insert(
            0,
            {
                "boxes_2d": boxes_2d[backdrop_inds],
                "scores_2d": scores_2d[backdrop_inds],
                "boxes_3d": boxes_3d[backdrop_inds],
                "scores_3d": scores_3d[backdrop_inds],
                "class_ids": class_ids[backdrop_inds],
                "embeddings": embeddings[backdrop_inds],
                "motion_models": backdrop_motion_model,
            },
        )

        # delete invalid tracks from memory
        invalid_ids = []
        for k, v in self.tracks.items():
            if frame_id - v["last_frame"] >= self.memory_limit:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

        if len(self.backdrops) > self.backdrop_memory_limit:
            self.backdrops.pop()

    def update_track(
        self,
        track_id: int,
        box_2d: Tensor,
        score_2d: Tensor,
        box_3d: Tensor,
        score_3d: Tensor,
        class_id: Tensor,
        embed: Tensor,
        obs_box_3d: Tensor,
        frame_id: int,
    ) -> None:
        """Update a track."""
        self.tracks[track_id]["box_2d"] = box_2d
        self.tracks[track_id]["score_2d"] = score_2d
        self.tracks[track_id]["motion_model"].update(obs_box_3d, score_3d)

        pd_box_3d = self.tracks[track_id]["motion_model"].get_state()[
            : self.motion_dims
        ]

        prev_obs = torch.cat(
            [
                self.tracks[track_id]["box_3d"][:6],
                self.tracks[track_id]["box_3d"][8].unsqueeze(0),
            ]
        )

        self.tracks[track_id]["box_3d"] = box_3d
        self.tracks[track_id]["box_3d"][:6] = pd_box_3d[:6]
        self.tracks[track_id]["box_3d"][8] = pd_box_3d[6]
        self.tracks[track_id]["box_3d"][9:12] = self.tracks[track_id][
            "motion_model"
        ].predict_velocity()
        self.tracks[track_id]["score_3d"] = score_3d
        self.tracks[track_id]["class_id"] = class_id

        self.tracks[track_id]["embed"] = (
            1 - self.memory_momentum
        ) * self.tracks[track_id]["embed"] + self.memory_momentum * embed

        velocity = (pd_box_3d - prev_obs) / (
            frame_id - self.tracks[track_id]["last_frame"]
        )

        self.tracks[track_id]["velocity"] = (
            self.tracks[track_id]["velocity"]
            * self.tracks[track_id]["acc_frame"]
            + velocity
        ) / (self.tracks[track_id]["acc_frame"] + 1)

        self.tracks[track_id]["last_frame"] = frame_id
        self.tracks[track_id]["acc_frame"] += 1

    def create_track(
        self,
        track_id: int,
        box_2d: Tensor,
        score_2d: Tensor,
        box_3d: Tensor,
        score_3d: Tensor,
        class_id: Tensor,
        embed: Tensor,
        obs_box_3d: Tensor,
        frame_id: int,
    ) -> None:
        """Create a new track."""
        motion_model = self.build_motion_model(obs_box_3d)

        self.tracks[track_id] = Track(
            box_2d=box_2d,
            score_2d=score_2d,
            box_3d=box_3d,
            score_3d=score_3d,
            class_id=class_id,
            embed=embed,
            motion_model=motion_model,
            velocity=torch.zeros(self.motion_dims, device=box_3d.device),
            last_frame=frame_id,
            acc_frame=0,
        )

    def build_motion_model(self, obs_3d: Tensor) -> BaseMotionModel:
        """Build motion model."""
        if self.motion_model == "KF3D":
            return KF3DMotionModel(
                num_frames=self.num_frames,
                obs_3d=obs_3d,
                motion_dims=self.motion_dims,
                fps=self.fps,
            )
        elif self.motion_model == "VeloLSTM":
            return LSTM3DMotionModel(
                num_frames=self.num_frames,
                lstm_model=self.lstm_model,
                obs_3d=obs_3d,
                motion_dims=self.motion_dims,
                fps=self.fps,
            )
        else:
            raise NotImplementedError(
                f"Motion model: {self.motion_model} not known!"
            )
