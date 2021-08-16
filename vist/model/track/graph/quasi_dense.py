"""Quasi-dense embedding similarity based graph."""
from typing import Dict, List, Optional, Tuple, Union

import torch
from pydantic import validator

from vist.common.bbox.utils import compute_iou
from vist.struct import Boxes2D

from .base import BaseTrackGraph, TrackGraphConfig


class QDTrackGraphConfig(TrackGraphConfig):
    """Quasi-dense similarity based graph config."""

    keep_in_memory: int  # threshold for keeping occluded objects in memory
    init_score_thr: float = 0.7
    obj_score_thr: float = 0.3
    match_score_thr: float = 0.5
    memo_backdrop_frames: int = 1
    memo_momentum: float = 0.8
    nms_conf_thr: float = 0.5
    nms_backdrop_iou_thr: float = 0.3
    nms_class_iou_thr: float = 0.7
    with_cats: bool = True

    @validator("memo_momentum", check_fields=False)
    def validate_memo_momentum(  # pylint: disable=no-self-argument,no-self-use,line-too-long
        cls, value: float
    ) -> float:
        """Check memo_momentum attribute."""
        if not 0 <= value <= 1.0:
            raise ValueError("memo_momentum must be >= 0 and <= 1.0")
        return value

    @validator("keep_in_memory", check_fields=False)
    def validate_keep_in_memory(  # pylint: disable=no-self-argument,no-self-use,line-too-long
        cls, value: int
    ) -> int:
        """Check keep_in_memory attribute."""
        if not value >= 0:
            raise ValueError("keep_in_memory must be >= 0")
        return value

    @validator("memo_backdrop_frames", check_fields=False)
    def validate_memo_backdrop_frames(  # pylint: disable=no-self-argument,no-self-use,line-too-long
        cls, value: int
    ) -> int:
        """Check memo_backdrop_frames attribute."""
        if not value >= 0:
            raise ValueError("memo_backdrop_frames must be >= 0")
        return value


class QDTrackGraph(BaseTrackGraph):
    """Tracking graph construction for quasi-dense instance similarity."""

    def __init__(self, cfg: TrackGraphConfig) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg = QDTrackGraphConfig(**cfg.dict())

    def reset(self) -> None:
        """Reset tracks."""
        super().reset()
        self.backdrops: List[Dict[str, Union[Boxes2D, torch.Tensor]]] = []

    def get_tracks(
        self,
        device: torch.device,
        frame_id: Optional[int] = None,
        add_backdrops: bool = False,
    ) -> Tuple[Boxes2D, torch.Tensor]:
        """Get active tracks at given frame.

        If frame_id is None, return all tracks in memory.
        """
        bboxs, embeds, cls, ids = [], [], [], []
        for k, v in self.tracks.items():
            if frame_id is None or v["last_frame"] == frame_id:
                bboxs.append(v["bbox"].unsqueeze(0))
                embeds.append(v["embed"].unsqueeze(0))
                cls.append(v["class_id"])
                ids.append(k)

        bboxs = (
            torch.cat(bboxs)
            if len(bboxs) > 0
            else torch.empty((0, 5), device=device)
        )
        embeds = (
            torch.cat(embeds)
            if len(embeds) > 0
            else torch.empty((0,), device=device)
        )
        cls = (
            torch.cat(cls)
            if len(cls) > 0
            else torch.empty((0,), device=device)
        )
        ids = torch.tensor(ids).to(device)

        if add_backdrops:
            for backdrop in self.backdrops:
                backdrop_ids = torch.full(
                    (len(backdrop["embeddings"]),),
                    -1,
                    dtype=torch.long,
                    device=device,
                )
                ids = torch.cat([ids, backdrop_ids])
                bboxs = torch.cat([bboxs, backdrop["detections"].boxes])
                embeds = torch.cat([embeds, backdrop["embeddings"]])
                cls = torch.cat([cls, backdrop["detections"].class_ids])

        return Boxes2D(bboxs, cls, ids), embeds

    def remove_duplicates(
        self, detections: Boxes2D, embeddings: torch.Tensor
    ) -> Tuple[Boxes2D, torch.Tensor]:
        """Remove overlapping objects across classes via nms."""
        # duplicate removal for potential backdrops and cross classes
        _, inds = detections.boxes[:, -1].sort(descending=True)
        detections, embeddings = detections[inds], embeddings[inds]
        valids = embeddings.new_ones((len(detections),))
        ious = compute_iou(detections, detections)
        for i in range(1, len(detections)):
            if detections.boxes[i, -1] < self.cfg.obj_score_thr:
                thr = self.cfg.nms_backdrop_iou_thr
            else:
                thr = self.cfg.nms_class_iou_thr

            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        detections = detections[valids, :]
        embeddings = embeddings[valids, :]
        return detections, embeddings

    def forward(  # type: ignore # pylint: disable=arguments-differ
        self, detections: Boxes2D, frame_id: int, embeddings: torch.Tensor
    ) -> Boxes2D:
        """Process inputs, match detections with existing tracks."""
        detections, embeddings = self.remove_duplicates(detections, embeddings)

        # init ids container
        ids = torch.full(
            (len(detections),), -1, dtype=torch.long, device=detections.device
        )

        # match if buffer is not empty
        if len(detections) > 0 and not self.empty:
            memo_dets, memo_embeds = self.get_tracks(
                detections.device, add_backdrops=True
            )

            # match using bisoftmax metric
            feats = torch.mm(embeddings, memo_embeds.t())
            d2t_scores = feats.softmax(dim=1)
            t2d_scores = feats.softmax(dim=0)
            scores = (d2t_scores + t2d_scores) / 2

            if self.cfg.with_cats:
                cat_same = detections.class_ids.view(
                    -1, 1
                ) == memo_dets.class_ids.view(1, -1)
                scores *= cat_same.float()

            for i in range(len(detections)):
                conf, memo_ind = torch.max(scores[i, :], dim=0)
                cur_id = memo_dets.track_ids[memo_ind]
                if conf > self.cfg.match_score_thr:
                    if cur_id > -1:
                        if detections.boxes[i, -1] > self.cfg.obj_score_thr:
                            ids[i] = cur_id
                            scores[:i, memo_ind] = 0
                            scores[(i + 1) :, memo_ind] = 0
                        elif conf > self.cfg.nms_conf_thr:  # pragma: no cover
                            ids[i] = -2
        new_inds = (ids == -1) & (
            detections.boxes[:, -1] > self.cfg.init_score_thr
        )
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracks,
            self.num_tracks + num_news,
            dtype=torch.long,
            device=ids.device,
        )
        self.num_tracks += num_news

        self.update(ids, detections, embeddings, frame_id)
        result, _ = self.get_tracks(detections.device, frame_id)
        return result

    def update(  # type: ignore # pylint: disable=arguments-differ
        self,
        ids: torch.Tensor,
        detections: Boxes2D,
        embeddings: torch.Tensor,
        frame_id: int,
    ) -> None:
        """Update track memory using matched detections."""
        tracklet_inds = ids > -1

        # update memo
        for cur_id, det, embed in zip(
            ids[tracklet_inds],
            detections[tracklet_inds],
            embeddings[tracklet_inds],
        ):  # type: ignore
            cur_id = int(cur_id)
            if cur_id in self.tracks.keys():
                self.update_track(cur_id, det, embed, frame_id)
            else:
                self.create_track(cur_id, det, embed, frame_id)

        backdrop_inds = torch.nonzero(ids == -1, as_tuple=False).squeeze(1)
        ious = compute_iou(detections[backdrop_inds], detections)
        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.cfg.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]

        self.backdrops.insert(
            0,
            dict(
                detections=detections[backdrop_inds],
                embeddings=embeddings[backdrop_inds],
            ),
        )

        # delete invalid tracks from memory
        invalid_ids = []
        for k, v in self.tracks.items():
            if frame_id - v["last_frame"] >= self.cfg.keep_in_memory:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

        if len(self.backdrops) > self.cfg.memo_backdrop_frames:
            self.backdrops.pop()

    def update_track(
        self,
        track_id: int,
        detection: Boxes2D,
        embedding: torch.Tensor,
        frame_id: int,
    ) -> None:
        """Update a specific track with a new models."""
        bbox, cls = detection.boxes[0], detection.class_ids[0]
        velocity = (bbox - self.tracks[track_id]["bbox"]) / (
            frame_id - self.tracks[track_id]["last_frame"]
        )
        self.tracks[track_id]["bbox"] = bbox
        self.tracks[track_id]["embed"] = (
            1 - self.cfg.memo_momentum
        ) * self.tracks[track_id]["embed"] + self.cfg.memo_momentum * embedding
        self.tracks[track_id]["last_frame"] = frame_id
        self.tracks[track_id]["class_id"] = cls
        self.tracks[track_id]["velocity"] = (
            self.tracks[track_id]["velocity"]
            * self.tracks[track_id]["acc_frame"]
            + velocity
        ) / (self.tracks[track_id]["acc_frame"] + 1)
        self.tracks[track_id]["acc_frame"] += 1

    def create_track(
        self,
        track_id: int,
        detection: Boxes2D,
        embedding: torch.Tensor,
        frame_id: int,
    ) -> None:
        """Create a new track from a models."""
        bbox, cls = detection.boxes[0], detection.class_ids[0]
        self.tracks[track_id] = dict(
            bbox=bbox,
            embed=embedding,
            class_id=cls,
            last_frame=frame_id,
            velocity=torch.zeros_like(bbox),
            acc_frame=0,
        )
