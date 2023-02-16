"""Memory for CC-3DT inference."""
from __future__ import annotations

import pdb
from typing import NamedTuple

import torch
from torch import Tensor

from vis4d.op.box.box2d import bbox_iou

from .util import update_frames, concat_states, get_last_tracks, merge_tracks


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
    velocities: Tensor
    last_frames: Tensor
    acc_frames: Tensor


class CC3DTrackMemory:
    """CC-3DT Track Memory."""

    def __init__(
        self,
        memory_limit: int = -1,
        nms_backdrop_iou_thr: float = 0.3,
        motion_dims: int = 7,
        backdrop_memory_limit: int = 1,
    ):
        """Creates an instance of the class."""
        assert memory_limit >= -1
        self.memory_limit = memory_limit
        self.motion_dims = motion_dims
        self.frames: list[CC3DTrackState] = []
        self.backdrop_frames: list[CC3DTrackState] = []
        assert backdrop_memory_limit >= 0
        self.backdrop_memory_limit = backdrop_memory_limit
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr

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

        self.frames = update_frames(self.frames, new_tracks, self.memory_limit)

        # backdrops
        backdrop_tracks = torch.nonzero(
            data.track_ids == -1, as_tuple=False
        ).squeeze(1)

        ious = bbox_iou(
            data.boxes[backdrop_tracks],
            data.boxes,
            data.camera_ids[backdrop_tracks],
            data.camera_ids,
        )

        for i, ind in enumerate(backdrop_tracks):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_tracks[i] = -1
        backdrop_tracks = backdrop_tracks[backdrop_tracks > -1]

        new_backdrops = CC3DTrackState(
            *(entry[backdrop_tracks] for entry in data)
        )
        self.backdrop_frames = update_frames(
            self.backdrop_frames, new_backdrops, self.backdrop_memory_limit
        )

    def replace_frame(
        self, frame_id: int, state_attr: str, state_value: Tensor
    ):
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
            track (list[CC3DTrackState]): List of track states for given query track.
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
        return track, frame_ids  # type: ignore

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
            velocities,
            last_frames,
            acc_frames,
        )

    def get_current_tracks(self, device: torch.device) -> CC3DTrackState:
        """Get all active tracks and backdrops in memory."""
        # get last states of all tracks
        if len(self.frames) > 0:
            memory_states = CC3DTrackState(*(concat_states(self.frames)))

            last_tracks = CC3DTrackState(
                *(get_last_tracks(memory_states, True))
            )
        else:
            last_tracks = self.get_empty_frame(0, device)

        # add backdrops
        if len(self.backdrop_frames) > 0:
            backdrops = CC3DTrackState(*(concat_states(self.backdrop_frames)))

            if backdrops.embeddings.size(1) != last_tracks.embeddings.size(1):
                assert (
                    len(last_tracks.embeddings) == 0
                ), "Unequal shape of backdrop embeddings and track embeddings!"
                last_frames = last_frames._replace(
                    embeddings=torch.empty(
                        (0, backdrops.embeddings.size(1)), device=device
                    )
                )

            last_tracks = CC3DTrackState(
                *(merge_tracks(last_tracks, backdrops))
            )
        return last_tracks
