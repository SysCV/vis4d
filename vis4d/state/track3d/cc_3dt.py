"""Memory for CC-3DT inference."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor, nn

from vis4d.common.typing import DictStrAny
from vis4d.op.box.box2d import bbox_iou
from vis4d.op.track3d.cc_3dt import CC3DTrackAssociation, get_track_3d_out
from vis4d.op.track3d.common import Track3DOut
from vis4d.op.track.assignment import TrackIDCounter

from .motion import BaseMotionModel, KF3DMotionModel, LSTM3DMotionModel


class Track(TypedDict):
    """CC-3DT Track state.

    Attributes:
        box_2d (Tensor): In shape (4,) and contains x1, y1, x2, y2.
        score_2d (Tensor): In shape (1,).
        box_3d (Tensor): In shape (12,) contains x,y,z,h,w,l,rx,ry,rz,vx,vy,vz.
        score_3d (Tensor): In shape (1,).
        class_id (Tensor): In shape (1,).
        embed (Tensor): In shape (E,). E is the embedding dimension.
        motion_model (BaseMotionModel): The motion model.
        velocity (Tensor): In shape (motion_dims,).
        last_frame (int): The last frame the track was updated.
        acc_frame (int): The number of frames the track was updated.
    """

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


class CC3DTrackGraph:
    """CC-3DT tracking graph."""

    def __init__(
        self,
        track: CC3DTrackAssociation | None = None,
        memory_size: int = 10,
        memory_momentum: float = 0.8,
        backdrop_memory_size: int = 1,
        nms_backdrop_iou_thr: float = 0.3,
        motion_model: str = "KF3D",
        lstm_model: nn.Module | None = None,
        motion_dims: int = 7,
        num_frames: int = 5,
        fps: int = 2,
        update_3d_score: bool = True,
        add_backdrops: bool = True,
    ) -> None:
        """Creates an instance of the class."""
        assert memory_size >= 0
        self.memory_size = memory_size
        assert 0 <= memory_momentum <= 1.0
        self.memory_momentum = memory_momentum
        assert backdrop_memory_size >= 0
        self.backdrop_memory_size = backdrop_memory_size
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr

        self.tracker = CC3DTrackAssociation() if track is None else track

        self.tracklets: dict[int, Track] = {}
        self.backdrops: list[DictStrAny] = []

        if motion_model == "VeloLSTM":
            assert (
                lstm_model is not None
            ), "lstm_model must be provided for VeloLSTM"
            self.lstm_model = lstm_model

        self.motion_model = motion_model
        self.motion_dims = motion_dims
        self.num_frames = num_frames
        self.fps = fps
        self.update_3d_score = update_3d_score
        self.add_backdrops = add_backdrops

    def reset(self) -> None:
        """Empty the memory."""
        self.tracklets.clear()
        self.backdrops.clear()

    def is_empty(self) -> bool:
        """Check if the memory is empty."""
        return len(self.tracklets) == 0

    def get_tracks(
        self,
        device: torch.device,
        frame_id: int | None = None,
        add_backdrops: bool = False,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        list[BaseMotionModel],
        Tensor,
    ]:
        """Get tracklests.

        If the frame_id is not provided, will return the latest state of all
        tracklets. Otherwise, will return the state of all tracklets at the
        given frame_id. If add_backdrops is True, will also return the
        backdrops.

        Args:
            device (torch.device): Device to put the tensors on.
            frame_id (int, optional): Frame id to query. Defaults to None.
            add_backdrops (bool, optional): Whether to add backdrops to the
                output. Defaults to False.

        Returns:
            boxes_2d (Tensor): 2D boxes in shape (N, 4).
            scores_2d (Tensor): 2D scores in shape (N,).
            boxes_3d (Tensor): 3D boxes in shape (N, 12).
            scores_3d (Tensor): 3D scores in shape (N,).
            class_ids (Tensor): Class ids in shape (N,).
            track_ids (Tensor): Track ids in shape (N,).
            embeds (Tensor): Embeddings in shape (N, E).
            motion_models (list[BaseMotionModel]): Motion models.
            velocities (Tensor): Velocities in shape (N, 3).
        """
        (
            boxes_2d_list,
            scores_2d_list,
            boxes_3d_list,
            scores_3d_list,
            class_ids_list,
            embeds_list,
            motion_models,
            velocities_list,
            track_ids_list,
        ) = ([], [], [], [], [], [], [], [], [])

        for track_id, track in self.tracklets.items():
            if frame_id is None or track["last_frame"] == frame_id:
                boxes_2d_list.append(track["box_2d"].unsqueeze(0))
                scores_2d_list.append(track["score_2d"].unsqueeze(0))
                boxes_3d_list.append(track["box_3d"].unsqueeze(0))
                scores_3d_list.append(track["score_3d"].unsqueeze(0))
                class_ids_list.append(track["class_id"].unsqueeze(0))
                embeds_list.append(track["embed"].unsqueeze(0))
                motion_models.append(track["motion_model"])
                velocities_list.append(track["velocity"].unsqueeze(0))
                track_ids_list.append(track_id)

        boxes_2d = (
            torch.cat(boxes_2d_list)
            if len(boxes_2d_list) > 0
            else torch.empty((0, 4), device=device)
        )
        scores_2d = (
            torch.cat(scores_2d_list)
            if len(scores_2d_list) > 0
            else torch.empty((0,), device=device)
        )
        boxes_3d = (
            torch.cat(boxes_3d_list)
            if len(boxes_3d_list) > 0
            else torch.empty((0, 12), device=device)
        )
        scores_3d = (
            torch.cat(scores_3d_list)
            if len(scores_3d_list) > 0
            else torch.empty((0,), device=device)
        )
        class_ids = (
            torch.cat(class_ids_list)
            if len(class_ids_list) > 0
            else torch.empty((0,), device=device)
        )
        embeds = (
            torch.cat(embeds_list)
            if len(embeds_list) > 0
            else torch.empty((0,), device=device)
        )
        velocities = (
            torch.cat(velocities_list)
            if len(velocities_list) > 0
            else torch.empty((0, self.motion_dims), device=device)
        )
        track_ids = torch.tensor(track_ids_list, device=device)

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
        """Update the tracker with new detections."""
        if frame_id == 0:
            self.reset()
            TrackIDCounter.reset()

        if not self.is_empty():
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
            ) = self.get_tracks(
                boxes_2d.device, add_backdrops=self.add_backdrops
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

        track_ids, filter_indices = self.tracker(
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
            self.update_3d_score,
        )

        self.update(
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
        ) = self.get_tracks(boxes_2d.device, frame_id=frame_id)

        # update 3D score
        if self.update_3d_score:
            track_scores_3d = scores_2d * scores_3d
        else:
            track_scores_3d = scores_3d

        return get_track_3d_out(
            boxes_3d, class_ids, track_scores_3d, track_ids
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
            if track_id in self.tracklets:
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
        for track_id, track in self.tracklets.items():
            if frame_id > track["last_frame"] and track_id > -1:
                pd_box_3d = track["motion_model"].predict()
                track["box_3d"][:6] = pd_box_3d[:6]
                track["box_3d"][8] = pd_box_3d[6]

        # Backdrops
        backdrop_inds = torch.nonzero(
            torch.eq(track_ids, -1), as_tuple=False
        ).squeeze(1)

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
        for k, v in self.tracklets.items():
            if frame_id - v["last_frame"] >= self.memory_size:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

        if len(self.backdrops) > self.backdrop_memory_size:
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
        self.tracklets[track_id]["box_2d"] = box_2d
        self.tracklets[track_id]["score_2d"] = score_2d
        self.tracklets[track_id]["motion_model"].update(obs_box_3d, score_3d)

        pd_box_3d = self.tracklets[track_id]["motion_model"].get_state()[
            : self.motion_dims
        ]

        prev_obs = torch.cat(
            [
                self.tracklets[track_id]["box_3d"][:6],
                self.tracklets[track_id]["box_3d"][8].unsqueeze(0),
            ]
        )

        self.tracklets[track_id]["box_3d"] = box_3d
        self.tracklets[track_id]["box_3d"][:6] = pd_box_3d[:6]
        self.tracklets[track_id]["box_3d"][8] = pd_box_3d[6]
        self.tracklets[track_id]["box_3d"][9:12] = self.tracklets[track_id][
            "motion_model"
        ].predict_velocity()
        self.tracklets[track_id]["score_3d"] = score_3d
        self.tracklets[track_id]["class_id"] = class_id

        self.tracklets[track_id]["embed"] = (
            1 - self.memory_momentum
        ) * self.tracklets[track_id]["embed"] + self.memory_momentum * embed

        velocity = (pd_box_3d - prev_obs) / (
            frame_id - self.tracklets[track_id]["last_frame"]
        )

        self.tracklets[track_id]["velocity"] = (
            self.tracklets[track_id]["velocity"]
            * self.tracklets[track_id]["acc_frame"]
            + velocity
        ) / (self.tracklets[track_id]["acc_frame"] + 1)

        self.tracklets[track_id]["last_frame"] = frame_id
        self.tracklets[track_id]["acc_frame"] += 1

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

        self.tracklets[track_id] = Track(
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

        if self.motion_model == "VeloLSTM":
            return LSTM3DMotionModel(
                num_frames=self.num_frames,
                lstm_model=self.lstm_model,
                obs_3d=obs_3d,
                motion_dims=self.motion_dims,
                fps=self.fps,
            )

        raise NotImplementedError(
            f"Motion model: {self.motion_model} not known!"
        )
