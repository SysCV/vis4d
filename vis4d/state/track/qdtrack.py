"""Memory for QDTrack inference."""
from __future__ import annotations

from typing import NamedTuple

import torch

from vis4d.op.box.box2d import bbox_iou

from ..util import concat_states, get_last_tracks, merge_tracks, update_frames


class QDTrackState(NamedTuple):
    """QDTrack Track state."""

    track_ids: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor
    class_ids: torch.Tensor
    embeddings: torch.Tensor


class QDTrackMemory:
    """QDTrack track memory.

    We store both tracks and backdrops here. The current active tracks are all
    tracks within the memory limit.
    """

    def __init__(
        self,
        memory_limit: int = -1,
        nms_backdrop_iou_thr: float = 0.3,
        backdrop_memory_limit: int = 1,
    ) -> None:
        """Creates an instance of the class.

        Args:
            memory_limit (int, optional): Maximum number of frames to be stored
                inside the memory. Defaults to -1.
            nms_backdrop_iou_thr (float, optional): IoU threshold for NMS
                between backdrops. Defaults to 0.3.
            backdrop_memory_limit (int, optional): Maximum number of frames
                backdrops are stored. Defaults to 1.
            memory_momentum (float, optional): Momentum value for accumulating
                embedding vectors across a track's memory buffer. Defaults
                to 0.8.
        """
        assert memory_limit >= -1
        self.memory_limit = memory_limit
        self.frames: list[QDTrackState] = []
        self.backdrop_frames: list[QDTrackState] = []
        assert backdrop_memory_limit >= 0
        self.backdrop_memory_limit = backdrop_memory_limit
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr

    def reset(self) -> None:
        """Empty the memory."""
        self.frames.clear()
        self.backdrop_frames.clear()

    def update(self, data: QDTrackState) -> None:
        """Update the track memory with a new state.

        Args:
            data (QDTrackState): The new state.
        """
        valid_tracks = torch.nonzero(
            data.track_ids > -1, as_tuple=False
        ).squeeze(1)

        new_tracks = QDTrackState(*(entry[valid_tracks] for entry in data))
        self.frames = update_frames(self.frames, new_tracks, self.memory_limit)

        # backdrops
        backdrop_tracks = torch.nonzero(
            data.track_ids == -1, as_tuple=False
        ).squeeze(1)

        ious = bbox_iou(data.boxes[backdrop_tracks], data.boxes)

        for i, ind in enumerate(backdrop_tracks):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_tracks[i] = -1

        backdrop_tracks = backdrop_tracks[backdrop_tracks > -1]
        new_backdrops = QDTrackState(
            *(entry[backdrop_tracks] for entry in data)
        )
        self.backdrop_frames = update_frames(
            self.backdrop_frames, new_backdrops, self.backdrop_memory_limit
        )

    def get_track(self, track_id: int) -> list[QDTrackState]:
        """Get representation of a single track across memory frames.

        Args:
            track_id (int): track id of query track.

        Returns:
            list[QDTrackState]: List of track states for given query track.
        """
        track = []
        for frame in self.frames:
            idx = (frame.track_ids == track_id).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                track.append(
                    QDTrackState(*(element[idx] for element in frame))
                )
        return track  # type: ignore

    def get_empty_frame(
        self, n_tracks: int, device: torch.device
    ) -> QDTrackState:
        """Get an empty frame with the correct number of tracks.

        Args:
            n_tracks (int): Number of tracks to allocate.
            device (torch.device): Device to allocate on.

        Returns:
            QDTrackState: Empty frame.
        """
        track_ids = torch.empty((n_tracks,), dtype=torch.int64, device=device)
        class_ids = torch.empty((n_tracks,), dtype=torch.int64, device=device)
        scores = torch.empty((n_tracks,), device=device)
        boxes = torch.empty((n_tracks, 4), device=device)
        embeddings = torch.empty((n_tracks, 1), device=device)
        return QDTrackState(track_ids, boxes, scores, class_ids, embeddings)

    def get_current_tracks(self, device: torch.device) -> QDTrackState:
        """Get all active tracks and backdrops in memory."""
        # get last states of all tracks
        if len(self.frames) > 0:
            memory = QDTrackState(*(concat_states(self.frames)))

            last_tracks = QDTrackState(*(get_last_tracks(memory)))
        else:
            last_tracks = self.get_empty_frame(0, device)

        # add backdrops
        if len(self.backdrop_frames) > 0:
            backdrops = QDTrackState(*(concat_states(self.backdrop_frames)))

            if backdrops.embeddings.size(1) != last_tracks.embeddings.size(1):
                assert (
                    len(last_tracks.embeddings) == 0
                ), "Unequal shape of backdrop embeddings and track embeddings!"
                last_tracks.embeddings = last_tracks._replace(
                    embeddings=torch.empty(
                        (0, backdrops.embeddings.size(1)), device=device
                    )
                )

            last_tracks = QDTrackState(*(merge_tracks(last_tracks, backdrops)))

        return last_tracks
