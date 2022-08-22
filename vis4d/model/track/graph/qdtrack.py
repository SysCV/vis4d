"""Quasi-dense embedding similarity based graph."""
import copy
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn

from vis4d.common.bbox.utils import bbox_iou
from vis4d.struct import Boxes2D, LabelInstances


class Tracks(NamedTuple):
    """Efficient tensor storage for QDTrack tracks."""



class Backdrops(NamedTuple):
    boxes: torch.Tensor
    scores: torch.Tensor
    embeddings: torch.Tensor
    class_ids: torch.Tensor



def bisoftmax_match(detection_class_ids: torch.Tensor, detection_embeddings: torch.Tensor, track_class_ids: torch.Tensor, track_embeddings: torch.Tensor, with_categories: bool = True) -> torch.Tensor:
        """match using bisoftmax metric"""
        feats = torch.mm(detection_embeddings, track_embeddings.t())
        d2t_scores = feats.softmax(dim=1)
        t2d_scores = feats.softmax(dim=0)
        similarity_scores = (d2t_scores + t2d_scores) / 2

        if with_categories:
            cat_same = detection_class_ids.view(
                -1, 1
            ) == track_class_ids.view(1, -1)
            similarity_scores *= cat_same.float()
        return similarity_scores


def greedy_assign(detection_scores: torch.Tensor, tracklet_ids: torch.Tensor, affinity_scores: torch.Tensor, num_existing_tracks: int) -> Tuple[torch.Tensor, int]:
    """Greedy assignment of detections to tracks given affinities."""
    ids = torch.full(
        (len(detection_scores),), -1, dtype=torch.long, device=detection_scores.device
    )

    for i in range(len(detection_scores)):
        breakpoint()
        conf, memo_ind = torch.max(affinity_scores[i, :], dim=0)
        cur_id = tracklet_ids[memo_ind]
        if conf > self.match_score_thr:
            if cur_id > -1:
                if detection_scores[i] > self.obj_score_thr:
                    ids[i] = cur_id
                    affinity_scores[:i, memo_ind] = 0
                    affinity_scores[(i + 1) :, memo_ind] = 0
                elif conf > self.nms_conf_thr:  # pragma: no cover
                    ids[i] = -2
    new_inds = (ids == -1) & (detection_scores > self.init_score_thr)
    num_new_tracks = new_inds.sum()
    ids[new_inds] = torch.arange(
        num_existing_tracks,
        num_existing_tracks + num_new_tracks,
        dtype=torch.long,
        device=ids.device,
    )
    return ids, num_existing_tracks + num_new_tracks


def filter_tracks(tracks: Tracks, keep_mask: torch.Tensor) -> Tracks:
    new_values = []
    for v in tracks:
         new_values.append(v[keep_mask])
    return Tracks(*new_values)


class BoxTracks:
    def __init__(self, device: torch.device):
        self.track_ids = []#torch.empty((0,), dtype=torch.long, device=device)
        self.boxes =
        self.scores =  #torch.empty((0,), dtype=torch.float, device=device)
        self.class_ids =  #torch.empty((0,), dtype=torch.long, device=device)

    def get(self, start_frame_index: int, end_frame_index: int) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        result = []
        for i in range(start_frame_index, end_frame_index):
            result.append((self.track_ids[i], self.boxes, self.scores, self.class_ids))
        return result

    @property
    def empty(self) -> bool:
        """Whether track memory is empty."""
        return len(self.tracks.track_ids) == 0

    def update(
        self,
        track_id: int,
        detections: torch.Tensor,
        class_ids: torch.Tensor,
        embeddings: torch.Tensor,
        frame_id: int,
    ) -> None:
        """Update a specific track with a new models."""

        velocity = (detections - self.tracks.boxes[tracklet_inds]) / (
            frame_id - self.tracks.last_frames[tracklet_inds]
        )
        self.tracks.boxes[tracklet_inds] = detection
        self.tracks[tracklet_inds].embed = (1 - self.memo_momentum) * self.tracks[
            track_id
        ].embed + self.memo_momentum * embedding
        self.tracks[track_id].last_frame = frame_id
        self.tracks[track_id].class_id = class_id
        self.tracks[track_id].velocity = (
            self.tracks[track_id].velocity * self.tracks[track_id].acc_frame
            + velocity
        ) / (self.tracks[track_id].acc_frame + 1)
        self.tracks[track_id].acc_frame += 1

    def create_tracks(
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


class Backdrop:

    def add_backdrops(self, ids, boxes, scores, class_ids) -> None:
        backdrop_inds = torch.nonzero(ids == -1, as_tuple=False).squeeze(1)
        ious = bbox_iou(boxes[backdrop_inds], boxes)  # TODO additional nms needed here?
        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]

        self.backdrops.insert(
            0,
            Backdrops(
                detections=detections[backdrop_inds],
                embeddings=embeddings[backdrop_inds],
            ),
        )
        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()



@torch.jit.script
class QDTrackGraph:
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

    def remove_duplicates(
        self,
        detections: torch.Tensor,
        scores: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def __call__(
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
        detection_scores = detection_scores[permute_inds][valids]
        detection_class_ids = detection_class_ids[permute_inds][valids]  # TODO move into remove_duplicates

        # match if buffer is not empty
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
            memo_track_ids = torch.cat([
                self.tracks.track_ids,
                self.tracks.track_ids.new_full((len(memo_embeds) - len(self.tracks.track_ids),), -1)
            ])

            # calculate affinities
            affinity_scores = bisoftmax_match(detection_class_ids, embeddings, memo_class_ids, memo_embeds)

            ids, self.num_tracks = greedy_assign(detection_scores, memo_track_ids, affinity_scores, self.num_tracks)
        else:
            ids = torch.full(
                (len(detection_scores),), -1, dtype=torch.long,
                device=detection_scores.device
            )

        self.update(ids, detections, embeddings, frame_id)
        return self.tracks

    def update(
        self,
        ids: torch.Tensor,
        detections: torch.Tensor,
        embeddings: torch.Tensor,
        frame_id: int,
    ) -> None:
        """Update track memory using matched detections."""
        # update existing tracks and add new ones
        update_track_ids = torch.nonzero(ids != -1, as_tuple=False).squeeze(1) # and in existing tracks
        new_track_ids = torch.nonzero(ids != -1, as_tuple=False).squeeze(1) # and not in existing tracks

        if cur_id in self.tracks:
            self.update_tracks(cur_id, det, embed, frame_id)
        else:
            self.create_tracks(cur_id, det, embed, frame_id)

        # delete invalid tracks from memory
        keep_mask = frame_id - self.tracks.last_frames < self.keep_in_memory
        self.tracks = filter_tracks(self.tracks, keep_mask)

