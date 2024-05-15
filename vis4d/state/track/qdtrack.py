"""Memory for QDTrack inference."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor

from vis4d.op.box.box2d import bbox_iou
from vis4d.op.track.assignment import TrackIDCounter
from vis4d.op.track.common import TrackOut
from vis4d.op.track.qdtrack import QDTrackAssociation


class Track(TypedDict):
    """QDTrack Track state.

    Attributes:
        box (Tensor): In shape (4,) and contains x1, y1, x2, y2.
        score (Tensor): In shape (1,).
        class_id (Tensor): In shape (1,).
        embedding (Tensor): In shape (E,). E is the embedding dimension.
        last_frame (int): Last frame id.
    """

    box: Tensor
    score: Tensor
    class_id: Tensor
    embed: Tensor
    last_frame: int


class QDTrackGraph:
    """Quasi-dense embedding similarity based graph."""

    def __init__(
        self,
        track: QDTrackAssociation | None = None,
        memory_size: int = 10,
        memory_momentum: float = 0.8,
        nms_backdrop_iou_thr: float = 0.3,
        backdrop_memory_size: int = 1,
    ) -> None:
        """Init."""
        assert memory_size >= 0
        self.memory_size = memory_size
        assert 0 <= memory_momentum <= 1.0
        self.memory_momentum = memory_momentum
        assert backdrop_memory_size >= 0
        self.backdrop_memory_size = backdrop_memory_size
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr

        self.tracker = QDTrackAssociation() if track is None else track

        self.tracklets: dict[int, Track] = {}
        self.backdrops: list[dict[str, Tensor]] = []

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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
            boxes (Tensor): 2D boxes in shape (N, 4).
            scores (Tensor): 2D scores in shape (N,).
            class_ids (Tensor): Class ids in shape (N,).
            track_ids (Tensor): Track ids in shape (N,).
            embeddings (Tensor): Embeddings in shape (N, E).
        """
        (
            boxes_list,
            scores_list,
            class_ids_list,
            embeddings_list,
            track_ids_list,
        ) = ([], [], [], [], [])

        for track_id, track in self.tracklets.items():
            if frame_id is None or track["last_frame"] == frame_id:
                boxes_list.append(track["box"].unsqueeze(0))
                scores_list.append(track["score"].unsqueeze(0))
                class_ids_list.append(track["class_id"].unsqueeze(0))
                embeddings_list.append(track["embed"].unsqueeze(0))
                track_ids_list.append(track_id)

        boxes = (
            torch.cat(boxes_list)
            if len(boxes_list) > 0
            else torch.empty((0, 4), device=device)
        )
        scores = (
            torch.cat(scores_list)
            if len(scores_list) > 0
            else torch.empty((0,), device=device)
        )
        class_ids = (
            torch.cat(class_ids_list)
            if len(class_ids_list) > 0
            else torch.empty((0,), device=device)
        )
        embeddings = (
            torch.cat(embeddings_list)
            if len(embeddings_list) > 0
            else torch.empty((0,), device=device)
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
                boxes = torch.cat([boxes, backdrop["boxes"]])
                scores = torch.cat([scores, backdrop["scores"]])
                class_ids = torch.cat([class_ids, backdrop["class_ids"]])
                embeddings = torch.cat([embeddings, backdrop["embeddings"]])

        return boxes, scores, class_ids, track_ids, embeddings

    def __call__(
        self,
        embeddings_list: list[Tensor],
        det_boxes_list: list[Tensor],
        det_scores_list: list[Tensor],
        class_ids_list: list[Tensor],
        frame_id_list: list[int],
    ) -> TrackOut:
        """Forward during test."""
        (
            batched_boxes,
            batched_scores,
            batched_class_ids,
            batched_track_ids,
        ) = ([], [], [], [])

        for frame_id, det_boxes, det_scores, class_ids, embeddings in zip(
            frame_id_list,
            det_boxes_list,
            det_scores_list,
            class_ids_list,
            embeddings_list,
        ):
            # reset graph at begin of sequence
            if frame_id == 0:
                self.reset()
                TrackIDCounter.reset()

            if not self.is_empty():
                (
                    _,
                    _,
                    memo_class_ids,
                    memo_track_ids,
                    memo_embeds,
                ) = self.get_tracks(det_boxes.device, add_backdrops=True)
            else:
                memo_class_ids = None
                memo_track_ids = None
                memo_embeds = None

            track_ids, filter_indices = self.tracker(
                det_boxes,
                det_scores,
                class_ids,
                embeddings,
                memo_track_ids,
                memo_class_ids,
                memo_embeds,
            )

            self.update(
                frame_id,
                track_ids,
                det_boxes[filter_indices],
                det_scores[filter_indices],
                class_ids[filter_indices],
                embeddings[filter_indices],
            )

            (
                boxes,
                scores,
                class_ids,
                track_ids,
                _,
            ) = self.get_tracks(det_boxes.device, frame_id=frame_id)

            batched_boxes.append(boxes)
            batched_scores.append(scores)
            batched_class_ids.append(class_ids)
            batched_track_ids.append(track_ids)

        return TrackOut(
            boxes=batched_boxes,
            class_ids=batched_class_ids,
            scores=batched_scores,
            track_ids=batched_track_ids,
        )

    def update(
        self,
        frame_id: int,
        track_ids: Tensor,
        boxes: Tensor,
        scores: Tensor,
        class_ids: Tensor,
        embeddings: Tensor,
    ) -> None:
        """Update the track memory with a new state."""
        valid_tracks = track_ids > -1

        # update memo
        for track_id, box, score, class_id, embed in zip(
            track_ids[valid_tracks],
            boxes[valid_tracks],
            scores[valid_tracks],
            class_ids[valid_tracks],
            embeddings[valid_tracks],
        ):
            track_id = int(track_id)
            if track_id in self.tracklets:
                self.update_track(
                    track_id, box, score, class_id, embed, frame_id
                )
            else:
                self.create_track(
                    track_id, box, score, class_id, embed, frame_id
                )

        # backdrops
        backdrop_inds = torch.nonzero(
            torch.eq(track_ids, -1), as_tuple=False
        ).squeeze(1)

        ious = bbox_iou(boxes[backdrop_inds], boxes)

        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]

        self.backdrops.insert(
            0,
            {
                "boxes": boxes[backdrop_inds],
                "scores": scores[backdrop_inds],
                "class_ids": class_ids[backdrop_inds],
                "embeddings": embeddings[backdrop_inds],
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
        box: Tensor,
        score: Tensor,
        class_id: Tensor,
        embedding: Tensor,
        frame_id: int,
    ) -> None:
        """Update a specific track with a new models."""
        self.tracklets[track_id]["box"] = box
        self.tracklets[track_id]["score"] = score
        self.tracklets[track_id]["class_id"] = class_id
        self.tracklets[track_id]["embed"] = (
            1 - self.memory_momentum
        ) * self.tracklets[track_id][
            "embed"
        ] + self.memory_momentum * embedding
        self.tracklets[track_id]["last_frame"] = frame_id

    def create_track(
        self,
        track_id: int,
        box: Tensor,
        score: Tensor,
        class_id: Tensor,
        embedding: Tensor,
        frame_id: int,
    ) -> None:
        """Create a new track from a models."""
        self.tracklets[track_id] = Track(
            box=box,
            score=score,
            class_id=class_id,
            embed=embedding,
            last_frame=frame_id,
        )
