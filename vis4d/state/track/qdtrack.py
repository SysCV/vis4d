"""Memory for QDTrack inference."""
from __future__ import annotations

from typing import NamedTuple

import torch

from .base import BaseTrackMemory


class QDTrackState(NamedTuple):
    """QDTrack Track state."""

    track_ids: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor
    class_ids: torch.Tensor
    embeddings: torch.Tensor


class QDTrackMemory(BaseTrackMemory[QDTrackState]):
    """QDTrack track memory.

    We store both tracks and backdrops here. The current active tracks are all
    tracks within the memory limit.
    """

    def __init__(
        self,
        memory_limit: int = -1,
        backdrop_memory_limit: int = 1,
        memory_momentum: float = 0.8,
    ):
        """Creates an instance of the class.

        Args:
            memory_limit (int, optional): Maximum number of frames to be stored
                inside the memory. Defaults to -1.
            backdrop_memory_limit (int, optional): Maximum number of frames
                backdrops are stored. Defaults to 1.
            memory_momentum (float, optional): Momentum value for accumulating
                embedding vectors across a track's memory buffer. Defaults
                to 0.8.
        """
        super().__init__(memory_limit)
        self.backdrop_frames: list[QDTrackState] = []
        self.memo_momentum = memory_momentum
        self.backdrop_memory_limit = backdrop_memory_limit
        assert backdrop_memory_limit >= 0
        assert 0 <= memory_momentum <= 1.0

    def reset(self) -> None:
        """Empty the memory."""
        super().reset()
        self.backdrop_frames.clear()

    def update(self, data: QDTrackState) -> None:
        """Update the track memory with a new state.

        Args:
            data (QDTrackState): The new state.
        """
        valid_tracks = data.track_ids != -1
        new_tracks = QDTrackState(*(entry[valid_tracks] for entry in data))
        new_backdrops = QDTrackState(*(entry[~valid_tracks] for entry in data))
        super().update(new_tracks)
        self.backdrop_frames.append(new_backdrops)
        if (
            self.backdrop_memory_limit >= 0
            and len(self.backdrop_frames) > self.backdrop_memory_limit
        ):
            self.backdrop_frames.pop(0)

    @staticmethod
    def _concat_states(states: list[QDTrackState]) -> QDTrackState:
        """Concatenate multiple states into a single one."""
        memory_track_ids = torch.cat(
            [mem_entry.track_ids for mem_entry in states]
        )
        memory_boxes = torch.cat([mem_entry.boxes for mem_entry in states])
        memory_scores = torch.cat([mem_entry.scores for mem_entry in states])
        memory_class_ids = torch.cat(
            [mem_entry.class_ids for mem_entry in states]
        )
        memory_embeddings = torch.cat(
            [mem_entry.embeddings for mem_entry in states]
        )
        return QDTrackState(
            memory_track_ids,
            memory_boxes,
            memory_scores,
            memory_class_ids,
            memory_embeddings,
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
                track.append(tuple(element[idx] for element in frame))
        return track  # type: ignore

    def get_current_tracks(self, device: torch.device) -> QDTrackState:
        """Get all active tracks and backdrops in memory."""
        # get last states of all tracks
        if len(self.frames) > 0:
            memory = self._concat_states(self.frames)

            track_ids = memory.track_ids.unique()
            class_ids = torch.zeros_like(track_ids)
            scores = torch.zeros(
                (
                    len(
                        track_ids,
                    )
                ),
                device=track_ids.device,
            )
            boxes = torch.zeros((len(track_ids), 4), device=track_ids.device)
            embeddings = torch.zeros(
                (len(track_ids), memory.embeddings.size(1)),
                device=track_ids.device,
            )

            # calculate exponential moving average of embedding across memory
            for i, track_id in enumerate(track_ids):
                track_mask = (memory.track_ids == track_id).nonzero(
                    as_tuple=True
                )[0]
                boxes[i] = memory.boxes[track_mask][-1]
                scores[i] = memory.scores[track_mask][-1]
                class_ids[i] = memory.class_ids[track_mask][-1]
                embeds = memory.embeddings[track_mask]
                embed = embeds[0]
                for mem_embed in embeds[1:]:
                    embed = (
                        1 - self.memo_momentum
                    ) * embed + self.memo_momentum * mem_embed
                embeddings[i] = embed
        else:
            track_ids = torch.empty((0,), dtype=torch.int64, device=device)
            class_ids = torch.empty((0,), dtype=torch.int64, device=device)
            scores = torch.empty((0,), device=device)
            boxes = torch.empty((0, 4), device=device)
            embeddings = torch.empty((0, 1), device=device)

        # add backdrops
        if len(self.backdrop_frames) > 0:
            backdrops = self._concat_states(self.backdrop_frames)
            track_ids = torch.cat([track_ids, backdrops.track_ids])
            boxes = torch.cat([boxes, backdrops.boxes])
            scores = torch.cat([scores, backdrops.scores])
            class_ids = torch.cat([class_ids, backdrops.class_ids])
            if backdrops.embeddings.size(1) != embeddings.size(1):
                assert (
                    len(embeddings) == 0
                ), "Unequal shape of backdrop embeddings and track embeddings!"
                embeddings = torch.empty(
                    (0, backdrops.embeddings.size(1)), device=device
                )
            embeddings = torch.cat([embeddings, backdrops.embeddings])

        return QDTrackState(track_ids, boxes, scores, class_ids, embeddings)
