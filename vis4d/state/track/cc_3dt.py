"""Memory for CC-3DT inference."""
from __future__ import annotations

import pdb
from typing import NamedTuple

import torch
from torch import Tensor

from vis4d.op.box.box2d import bbox_iou

from .base import BaseTrackMemory


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


class CC3DTrackMemory(BaseTrackMemory[CC3DTrackState]):
    """CC-3DT Track Memory."""

    def __init__(
        self,
        memory_limit: int = -1,
        nms_backdrop_iou_thr: float = 0.3,
        motion_dims: int = 7,
        backdrop_memory_limit: int = 1,
        memory_momentum: float = 0.8,
    ):
        """Creates an instance of the class."""
        super().__init__(memory_limit)
        self.backdrop_frames: list[CC3DTrackState] = []
        self.memo_momentum = memory_momentum
        self.backdrop_memory_limit = backdrop_memory_limit
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.motion_dims = motion_dims
        assert backdrop_memory_limit >= 0
        assert 0 <= memory_momentum <= 1.0

    def reset(self) -> None:
        """Empty the memory."""
        super().reset()
        self.backdrop_frames.clear()

    def update(self, data: CC3DTrackState) -> None:
        """Update the track memory with a new state."""
        valid_tracks = torch.nonzero(
            data.track_ids > -1, as_tuple=False
        ).squeeze(1)

        new_tracks = CC3DTrackState(*(entry[valid_tracks] for entry in data))

        super().update(new_tracks)

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
        self.backdrop_frames.append(new_backdrops)
        if (
            self.backdrop_memory_limit >= 0
            and len(self.backdrop_frames) > self.backdrop_memory_limit
        ):
            self.backdrop_frames.pop(0)

    def update_frame(self, frame_id: int, state_attr: str, value: Tensor):
        """Update the track memory with a new state."""
        self.frames[frame_id] = self.frames[frame_id]._replace(
            **{state_attr: value}
        )

    @staticmethod
    def _concat_states(states: list[CC3DTrackState]) -> CC3DTrackState:
        """Concatenate multiple states into a single one."""
        memory_track_ids = torch.cat(
            [mem_entry.track_ids for mem_entry in states]
        )
        memory_boxes = torch.cat([mem_entry.boxes for mem_entry in states])
        memory_camera_ids = torch.cat(
            [mem_entry.camera_ids for mem_entry in states]
        )
        memory_scores = torch.cat([mem_entry.scores for mem_entry in states])
        memory_boxes_3d = torch.cat(
            [mem_entry.boxes_3d for mem_entry in states]
        )
        memory_scores_3d = torch.cat(
            [mem_entry.scores_3d for mem_entry in states]
        )
        memory_class_ids = torch.cat(
            [mem_entry.class_ids for mem_entry in states]
        )
        memory_embeddings = torch.cat(
            [mem_entry.embeddings for mem_entry in states]
        )
        memory_motion_states = torch.cat(
            [mem_entry.motion_states for mem_entry in states]
        )
        memory_motion_hidden = torch.cat(
            [mem_entry.motion_hidden for mem_entry in states]
        )
        memory_velocities = torch.cat(
            [mem_entry.velocities for mem_entry in states]
        )
        memory_last_frames = torch.cat(
            [mem_entry.last_frames for mem_entry in states]
        )
        memory_acc_frames = torch.cat(
            [mem_entry.acc_frames for mem_entry in states]
        )
        return CC3DTrackState(
            memory_track_ids,
            memory_boxes,
            memory_camera_ids,
            memory_scores,
            memory_boxes_3d,
            memory_scores_3d,
            memory_class_ids,
            memory_embeddings,
            memory_motion_states,
            memory_motion_hidden,
            memory_velocities,
            memory_last_frames,
            memory_acc_frames,
        )

    def get_track(
        self, track_id: int
    ) -> tuple[list[CC3DTrackState], list[int]]:
        """Get representation of a single track across memory frames.

        Args:
            track_id (int): track id of query track.

        Returns:
            list[QDTrackState]: List of track states for given query track.
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

    def get_current_tracks(self, device: torch.device) -> CC3DTrackState:
        """Get all active tracks and backdrops in memory."""
        # get last states of all tracks
        if len(self.frames) > 0:
            memory = self._concat_states(self.frames)

            track_ids = memory.track_ids.unique()
            class_ids = torch.zeros_like(track_ids)
            camera_ids = torch.zeros(
                (
                    len(
                        track_ids,
                    )
                ),
                device=track_ids.device,
            )
            scores = torch.zeros(
                (
                    len(
                        track_ids,
                    )
                ),
                device=track_ids.device,
            )
            scores_3d = torch.zeros(
                (
                    len(
                        track_ids,
                    )
                ),
                device=track_ids.device,
            )
            boxes = torch.zeros((len(track_ids), 4), device=track_ids.device)
            boxes_3d = torch.zeros(
                (len(track_ids), 12), device=track_ids.device
            )
            embeddings = torch.zeros(
                (len(track_ids), memory.embeddings.size(1)),
                device=track_ids.device,
            )
            motion_states = torch.zeros(
                (len(track_ids), self.motion_dims + 3), device=track_ids.device
            )
            motion_hidden = torch.zeros(
                (len(track_ids), self.motion_dims + 3, self.motion_dims + 3),
                device=track_ids.device,
            )
            velocities = torch.zeros(
                (len(track_ids), self.motion_dims), device=track_ids.device
            )
            last_frames = torch.zeros(
                (len(track_ids)), device=track_ids.device
            )
            acc_frames = torch.zeros((len(track_ids)), device=track_ids.device)

            for i, track_id in enumerate(track_ids):
                track_mask = (memory.track_ids == track_id).nonzero(
                    as_tuple=False
                )[-1]
                boxes[i] = memory.boxes[track_mask]
                camera_ids[i] = memory.camera_ids[track_mask]
                scores[i] = memory.scores[track_mask]
                boxes_3d[i] = memory.boxes_3d[track_mask]
                scores_3d[i] = memory.scores_3d[track_mask]
                class_ids[i] = memory.class_ids[track_mask]
                embeds = memory.embeddings[track_mask]
                embed = embeds[0]
                embeddings[i] = embed
                motion_states[i] = memory.motion_states[track_mask]
                motion_hidden[i] = memory.motion_hidden[track_mask]
                velocities[i] = memory.velocities[track_mask]
                last_frames[i] = memory.last_frames[track_mask]
                acc_frames[i] = memory.acc_frames[track_mask]
        else:
            track_ids = torch.empty((0,), dtype=torch.int64, device=device)
            class_ids = torch.empty((0,), dtype=torch.int64, device=device)
            camera_ids = torch.empty((0,), dtype=torch.int64, device=device)
            scores = torch.empty((0,), device=device)
            boxes = torch.empty((0, 4), device=device)
            scores_3d = torch.empty((0,), device=device)
            boxes_3d = torch.empty((0, 12), device=device)
            embeddings = torch.empty((0, 1), device=device)
            motion_states = torch.empty(
                (0, self.motion_dims + 3), device=device
            )
            motion_hidden = torch.empty(
                (0, self.motion_dims + 3, self.motion_dims + 3), device=device
            )
            velocities = torch.empty((0, self.motion_dims), device=device)
            last_frames = torch.empty((0,), device=device)
            acc_frames = torch.empty((0,), device=device)

        # add backdrops
        if len(self.backdrop_frames) > 0:
            backdrops = self._concat_states(self.backdrop_frames)
            track_ids = torch.cat([track_ids, backdrops.track_ids])
            boxes = torch.cat([boxes, backdrops.boxes])
            camera_ids = torch.cat([camera_ids, backdrops.camera_ids])
            scores = torch.cat([scores, backdrops.scores])
            boxes_3d = torch.cat([boxes_3d, backdrops.boxes_3d])
            scores_3d = torch.cat([scores_3d, backdrops.scores_3d])
            class_ids = torch.cat([class_ids, backdrops.class_ids])
            if backdrops.embeddings.size(1) != embeddings.size(1):
                assert (
                    len(embeddings) == 0
                ), "Unequal shape of backdrop embeddings and track embeddings!"
                embeddings = torch.empty(
                    (0, backdrops.embeddings.size(1)), device=device
                )
            embeddings = torch.cat([embeddings, backdrops.embeddings])
            motion_states = torch.cat([motion_states, backdrops.motion_states])
            motion_hidden = torch.cat([motion_hidden, backdrops.motion_hidden])
            velocities = torch.cat([velocities, backdrops.velocities])
            last_frames = torch.cat([last_frames, backdrops.last_frames])
            acc_frames = torch.cat([acc_frames, backdrops.acc_frames])

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
