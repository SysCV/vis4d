"""Quasi-dense embedding similarity based tracker."""
import torch
import torch.nn.functional as F
from pydantic import validator

from openmt.structures import Boxes2D

from .base_tracker import BaseTracker, TrackLogicConfig


class QDEmbeddingTrackerConfig(TrackLogicConfig):
    assign_strategy: str  # e.g. greedy or hungarian
    keep_in_memory: int  # threshold for keeping occluded objects in memory
    init_score_thr: float = 0.8
    obj_score_thr: float = 0.5
    match_score_thr: float = 0.5
    memo_backdrop_frames: int = 1
    memo_momentum: float = 0.8
    nms_conf_thr: float = 0.5
    nms_backdrop_iou_thr: float = 0.3
    nms_class_iou_thr: float = 0.7
    with_cats: bool = True
    match_metric: str = "bisoftmax"

    @validator("memo_momentum", check_fields=False)
    def memo_momentum(cls, v):
        if not 0 <= v <= 1.0:
            raise ValueError("memo_momentum must be >= 0 and <= 1.0")
        return v

    @validator("memo_tracklet_frames", check_fields=False)
    def memo_tracklet_frames(cls, v):
        if not v >= 0:
            raise ValueError("memo_tracklet_frames must be >= 0")
        return v

    @validator("memo_backdrop_frames", check_fields=False)
    def memo_backdrop_frames(cls, v):
        if not v >= 0:
            raise ValueError("memo_backdrop_frames must be >= 0")
        return v

    @validator("match_metric", check_fields=False)
    def match_metric(cls, v):
        if not v in ["bisoftmax", "softmax", "cosine"]:
            raise ValueError(
                "match_metric must be in [bisoftmax, softmax, cosine]"
            )
        return v


class QDEmbeddingTracker(BaseTracker):
    """Embedding-based tracker for quasi-dense instance similarity."""

    def __init__(self, cfg: TrackLogicConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg = QDEmbeddingTrackerConfig(**cfg.__dict__)

    def forward(
        self, detections: Boxes2D, embeddings: torch.Tensor, frame_id: int
    ) -> None:  # TODO adapt to own data structures
        """Process inputs."""
        _, inds = detections.boxes[:, -1].sort(descending=True)
        detections = detections[inds, :]
        embeddings = embeddings[inds, :]

        # duplicate removal for potential backdrops and cross classes
        valids = embeddings.new_ones((len(detections),))
        ious = bbox_overlaps(
            detections, detections
        )  # TODO nms still necessary? else use matcher
        for i in range(1, len(detections)):
            if detections.boxes[i, -1] < self.obj_score_thr:
                thr = self.nms_backdrop_iou_thr
            else:
                thr = self.nms_class_iou_thr

            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        detections = detections[valids, :]
        embeddings = embeddings[valids, :]

        # init ids container
        ids = torch.full((len(detections),), -1, dtype=torch.long)

        # match if buffer is not empty
        if len(detections) > 0 and not self.empty:
            (
                memo_bboxes,
                memo_labels,
                memo_embeds,
                memo_ids,
                memo_vs,
            ) = self.memo

            if self.match_metric == "bisoftmax":
                feats = torch.mm(embeddings, memo_embeds.t())
                d2t_scores = feats.softmax(dim=1)
                t2d_scores = feats.softmax(dim=0)
                scores = (d2t_scores + t2d_scores) / 2
            elif self.match_metric == "softmax":
                feats = torch.mm(embeddings, memo_embeds.t())
                scores = feats.softmax(dim=1)
            elif self.match_metric == "cosine":
                scores = torch.mm(
                    F.normalize(embeddings, p=2, dim=1),
                    F.normalize(memo_embeds, p=2, dim=1).t(),
                )
            else:
                raise NotImplementedError

            if self.with_cats:
                cat_same = detections.classes.view(-1, 1) == memo_labels.view(
                    1, -1
                )
                scores *= cat_same.float()

            for i in range(len(detections)):
                conf, memo_ind = torch.max(scores[i, :], dim=0)
                id = memo_ids[memo_ind]
                if conf > self.match_score_thr:
                    if id > -1:
                        if detections.boxes[i, -1] > self.obj_score_thr:
                            ids[i] = id
                            scores[:i, memo_ind] = 0
                            scores[i + 1 :, memo_ind] = 0
                        else:
                            if conf > self.nms_conf_thr:
                                ids[i] = -2
        new_inds = (ids == -1) & (
            detections.boxes[i, -1] > self.init_score_thr
        ).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracklets, self.num_tracklets + num_news, dtype=torch.long
        )
        self.num_tracklets += num_news

        self.update_memo(ids, detections, embeddings, frame_id)

    def update(
        self, ids: torch.Tensor, detections, embeddings, frame_id
    ) -> None:
        """Update track memory."""
        tracklet_inds = ids > -1

        # update memo
        for id, bbox, label, embed in zip(
            ids[tracklet_inds],
            detections.boxes[tracklet_inds],
            detections.classes[tracklet_inds],
            embeddings[tracklet_inds],
        ):
            id = int(id)
            if id in self.tracklets.keys():
                velocity = (bbox - self.tracklets[id]["bbox"]) / (
                    frame_id - self.tracklets[id]["last_frame"]
                )
                self.tracklets[id]["bbox"] = bbox
                self.tracklets[id]["embed"] = (
                    1 - self.memo_momentum
                ) * self.tracklets[id]["embed"] + self.memo_momentum * embed
                self.tracklets[id]["last_frame"] = frame_id
                self.tracklets[id]["label"] = label
                self.tracklets[id]["velocity"] = (
                    self.tracklets[id]["velocity"]
                    * self.tracklets[id]["acc_frame"]
                    + velocity
                ) / (self.tracklets[id]["acc_frame"] + 1)
                self.tracklets[id]["acc_frame"] += 1
            else:
                self.tracklets[id] = dict(
                    bbox=bbox,
                    embed=embed,
                    label=label,
                    last_frame=frame_id,
                    velocity=torch.zeros_like(bbox),
                    acc_frame=0,
                )

        backdrop_inds = torch.nonzero(ids == -1, as_tuple=False).squeeze(1)
        ious = bbox_overlaps(
            detections[backdrop_inds], detections
        )  # TODO use matcher
        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]

        self.backdrops.insert(  # TODO implement backdrop buffer
            0,
            dict(
                detections=detections[backdrop_inds],
                embeddings=embeddings[backdrop_inds],
            ),
        )

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if frame_id - v["last_frame"] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()
