"""Quasi-dense embedding similarity based graph."""
import copy
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn

from vis4d.common.bbox.utils import bbox_iou
from vis4d.struct import Boxes2D, LabelInstances


class Tracks(NamedTuple):
    """Efficient tensor storage for QDTrack tracks."""

    track_ids: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor
    embeddings: torch.Tensor
    class_ids: torch.Tensor
    last_frames: torch.Tensor
    velocities: torch.Tensor
    acc_frames: torch.Tensor


class Backdrops(NamedTuple):
    boxes: torch.Tensor
    scores: torch.Tensor
    embeddings: torch.Tensor
    class_ids: torch.Tensor


class QDTrackGraph(nn.Module):
    """Tracking graph for quasi-dense instance similarity.

    Attributes:
        keep_in_memory: threshold for keeping occluded objects in memory
        init_score_thr: Confidence threshold for initializing a new track
        obj_score_thr: Confidence treshold s.t. a detection is considered in
        the track / det matching process.
        match_score_thr: Similarity score threshold for matching a detection to
        an existing track.
        memo_backdrop_frames: Number of timesteps to keep backdrops.
        memo_momentum: Momentum of embedding memory for smoothing embeddings.
        nms_conf_thr:
        nms_backdrop_iou_thr: Maximum IoU of a backdrop with another detection.
        nms_class_iou_thr: Maximum IoU of a high score detection with another
        of a different class.
        with_cats: If to consider category information for tracking (i.e. all
        detections within a track must have consistent category labels).

        Note: Backdrops are low-score detections kept in case they have high
        similarity with a high-score detection in succeeding frames.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        keep_in_memory: int = 10,
        init_score_thr: float = 0.7,
        obj_score_thr: float = 0.3,
        match_score_thr: float = 0.5,
        memo_backdrop_frames: int = 1,
        memo_momentum: float = 0.8,
        nms_conf_thr: float = 0.5,
        nms_backdrop_iou_thr: float = 0.3,
        nms_class_iou_thr: float = 0.7,
        with_cats: bool = True,
    ) -> None:
        """Init."""
        super().__init__()
        self.keep_in_memory = keep_in_memory
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_backdrop_frames = memo_backdrop_frames
        self.memo_momentum = memo_momentum
        self.nms_conf_thr = nms_conf_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.with_cats = with_cats
        self.embed_dim = 256  # TODO

        # validate arguments
        assert 0 <= memo_momentum <= 1.0
        assert keep_in_memory >= 0
        assert memo_backdrop_frames >= 0
        self.reset()

    def reset(self) -> None:
        """Reset tracks."""
        self.num_tracks = 0  # TODO device
        self.tracks = Tracks(
            track_ids=torch.empty((0,)),
            boxes=torch.empty((0, 4)),
            scores=torch.empty((0,)),
            embeddings=torch.empty((0, self.embed_dim)),
            class_ids=torch.empty((0,)),
            last_frames=torch.empty((0,)),
            velocities=torch.empty((0, 4)),
            acc_frames=torch.empty((0,)),
        )
        self.backdrops = [
            Backdrops(
                boxes=torch.empty((0, 4)),
                scores=torch.empty((0,)),
                embeddings=torch.empty((0, self.embed_dim)),
                class_ids=torch.empty((0,)),
            )
        ]

    def remove_duplicates(
        self,
        detections: torch.Tensor,
        scores: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> Tuple[Boxes2D, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove overlapping objects across classes via nms."""
        # duplicate removal for potential backdrops and cross classes
        scores, inds = scores.sort(descending=True)
        detections, embeddings = detections[inds], embeddings[inds]
        valids = embeddings.new_ones((len(detections),))
        ious = bbox_iou(detections, detections)
        for i in range(1, len(detections)):
            if scores[i] < self.obj_score_thr:
                thr = self.nms_backdrop_iou_thr
            else:
                thr = self.nms_class_iou_thr

            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        detections = detections[valids, :]
        embeddings = embeddings[valids, :]
        return detections, embeddings, valids, inds

    @property
    def empty(self) -> bool:
        """Whether track memory is empty."""
        return not self.tracks

    def forward(
        self,
        detections: torch.Tensor,
        detection_scores: torch.Tensor,
        detection_class_ids: torch.Tensor,
        embeddings: torch.Tensor,
        frame_id: int,
    ) -> Tracks:
        """Process inputs, match detections with existing tracks."""
        # reset graph at begin of sequence
        if frame_id == 0:
            self.reset()

        detections, embeddings, valids, permute_inds = self.remove_duplicates(
            detections, detection_scores, embeddings
        )

        # init ids container
        ids = torch.full(
            (len(detections),), -1, dtype=torch.long, device=detections.device
        )

        # match if buffer is not empty
        detections_scores = detections.score
        assert detections_scores is not None
        if len(detections) > 0 and not self.empty:
            memo_class_ids = torch.cat(
                [
                    self.tracks.class_ids,
                    *[bd.class_ids for bd in self.backdrops],
                ]
            )
            memo_embeds = torch.cat(
                [
                    self.tracks.embeddings,
                    *[bd.embeddings for bd in self.backdrops],
                ]
            )

            # match using bisoftmax metric
            feats = torch.mm(embeddings, memo_embeds.t())
            d2t_scores = feats.softmax(dim=1)
            t2d_scores = feats.softmax(dim=0)
            similarity_scores = (d2t_scores + t2d_scores) / 2

            if self.with_cats:
                cat_same = detections.class_ids.view(
                    -1, 1
                ) == memo_class_ids.view(1, -1)
                similarity_scores *= cat_same.float()

            for i in range(len(detections)):  # TODO this loop can be removed
                conf, memo_ind = torch.max(similarity_scores[i, :], dim=0)
                cur_id = memo_dets.track_ids[memo_ind]
                if conf > self.match_score_thr:
                    if cur_id > -1:
                        if detections_scores[i] > self.obj_score_thr:
                            ids[i] = cur_id
                            similarity_scores[:i, memo_ind] = 0
                            similarity_scores[(i + 1) :, memo_ind] = 0
                        elif conf > self.nms_conf_thr:  # pragma: no cover
                            ids[i] = -2
        new_inds = (ids == -1) & (detections_scores > self.init_score_thr)
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracks,
            self.num_tracks + num_news,
            dtype=torch.long,
            device=ids.device,
        )
        self.num_tracks += num_news

        self.update(ids, detections, embeddings, frame_id)

        valids[valids.clone()] = ids > -1  # remove backdrops, low score
        result = copy.deepcopy(predictions)
        for pred in result.get_instance_labels():
            if len(pred[0]) > 0:  # type: ignore
                pred[0] = pred[0][permute_inds][valids]  # type: ignore
                pred[0].track_ids = ids[ids > -1]  # type: ignore

        return result

    def update(
        self,
        ids: torch.Tensor,
        detections: torch.Tensor,
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
        ):
            cur_id = int(cur_id)
            if cur_id in self.tracks:
                self.update_track(cur_id, det, embed, frame_id)
            else:
                self.create_track(cur_id, det, embed, frame_id)

        backdrop_inds = torch.nonzero(ids == -1, as_tuple=False).squeeze(1)
        ious = bbox_iou(detections[backdrop_inds], detections)
        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
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
            if frame_id - v.last_frame >= self.keep_in_memory:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()

    def update_track(
        self,
        track_id: int,
        detection: torch.Tensor,
        class_id: torch.Tensor,
        embedding: torch.Tensor,
        frame_id: int,
    ) -> None:
        """Update a specific track with a new models."""
        velocity = (detection - self.tracks[track_id].bbox) / (
            frame_id - self.tracks[track_id].last_frame
        )
        self.tracks[track_id].bbox = detection
        self.tracks[track_id].embed = (1 - self.memo_momentum) * self.tracks[
            track_id
        ].embed + self.memo_momentum * embedding
        self.tracks[track_id].last_frame = frame_id
        self.tracks[track_id].class_id = class_id
        self.tracks[track_id].velocity = (
            self.tracks[track_id].velocity * self.tracks[track_id].acc_frame
            + velocity
        ) / (self.tracks[track_id].acc_frame + 1)
        self.tracks[track_id].acc_frame += 1

    def create_track(
        self,
        track_id: int,
        detection: torch.Tensor,
        class_id: torch.Tensor,
        embedding: torch.Tensor,
        frame_id: int,
    ) -> None:
        """Create a new track from a models."""
        self.tracks[track_id] = Track(
            bbox=detection,
            embed=embedding,
            class_id=class_id,
            last_frame=frame_id,
            velocity=torch.zeros_like(detection),
            acc_frame=0,
        )
