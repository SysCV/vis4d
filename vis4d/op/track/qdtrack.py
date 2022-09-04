"""Quasi-dense embedding similarity based graph."""
import copy
from typing import Generic, List, NamedTuple, Tuple, TypeVar

import torch

from vis4d.common.bbox.utils import bbox_iou

from .assignment import greedy_assign, random_ids
from .matching import calc_bisoftmax_affinity


class BaseTrackState(NamedTuple):
    """Basic Track state."""

    track_ids: torch.Tensor


TTrackState = TypeVar("TTrackState", bound=NamedTuple)


class BaseTrackMemory(Generic[TTrackState]):
    """Basic Track Memory class.

    Holds track representation across timesteps represented as:
    List[Tuple[torch.Tensor, ...]]
    where each list element is tracks at time t and tracks at time t are
    represented as Tuple[Tensor], where first element is a LongTensor of ids,
    and other N elements are boxes, scores, class_ids, etc.
    """

    def __init__(self, memory_limit: int = -1):
        assert memory_limit >= -1
        self.memory_limit = memory_limit
        self.frames: List[TTrackState] = []

    def reset(self) -> None:
        """Empty the memory."""
        self.frames: List[TTrackState] = []

    @property
    def last_frame(self) -> TTrackState:
        """Return last frame stored in memory.

        Returns:
            Tuple[torch.Tensor, ...]: Last frame representation.
        """
        return self.frames[-1]

    def get_frame(self, index: int) -> TTrackState:
        """Get TrackState at frame with given index."""
        return self.frames[index]

    def get_frames(
        self, start_index: int, end_index: int
    ) -> List[TTrackState]:
        """_summary_

        Args:
            start_index (int): _description_
            end_index (int): _description_

        Returns:
            List[Tuple[torch.Tensor, ...]]: _description_
        """
        return self.frames[start_index:end_index]

    def get_current_tracks(self) -> TTrackState:
        """Return active tracks."""
        return self.frames[-1]

    def get_track(self, track_id: int) -> List[Tuple[torch.Tensor, ...]]:
        """_summary_

        Args:
            track_id (int): _description_

        Returns:
            List[Tuple[torch.Tensor, ...]]: _description_
        """
        track = []
        for frame in self.frames:
            idx = (frame.track_ids == track_id).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                track.append(tuple(element[idx] for element in frame))
        return track

    def update(self, data: TTrackState) -> None:
        """Store valid tracks (id != -1) in memory."""
        self.frames.append(data)


class QDTrackState(NamedTuple):
    """QDTrack Track state."""

    track_ids: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor
    class_ids: torch.Tensor
    embeddings: torch.Tensor


class QDTrackMemory(BaseTrackMemory[QDTrackState]):
    """QDTrack track memory."""

    def __init__(
        self,
        memory_limit: int = -1,
        backdrop_memory_limit: int = 1,
        memory_momentum: float = 0.8,
    ):
        super().__init__(memory_limit)
        self.backdrop_frames: List[QDTrackState] = []
        self.memo_momentum = memory_momentum
        self.backdrop_memory_limit = backdrop_memory_limit
        assert backdrop_memory_limit >= 0
        assert 0 <= memory_momentum <= 1.0

    def reset(self) -> None:
        """Empty the memory."""
        super().reset()
        self.backdrop_frames: List[QDTrackState] = []

    def update(self, data: QDTrackState) -> None:
        valid_tracks = data.track_ids != -1
        new_tracks = QDTrackState(*(entry[valid_tracks] for entry in data))
        new_backdrops = QDTrackState(*(entry[~valid_tracks] for entry in data))
        super().update(new_tracks)
        self.backdrop_frames.append(new_backdrops)

    def get_current_tracks(self, device: torch.device) -> QDTrackState:
        """Get active tracks and backdrops."""
        if len(self.frames) > 0:
            memory_track_ids = torch.cat(
                [mem_entry.track_ids for mem_entry in self.frames]
            )
            memory_boxes = torch.cat(
                [mem_entry.boxes for mem_entry in self.frames]
            )
            memory_scores = torch.cat(
                [mem_entry.scores for mem_entry in self.frames]
            )
            memory_class_ids = torch.cat(
                [mem_entry.class_ids for mem_entry in self.frames]
            )
            memory_embeddings = torch.cat(
                [mem_entry.embeddings for mem_entry in self.frames]
            )

            all_track_ids = memory_track_ids.unique()
            all_class_ids = torch.zeros_like(all_track_ids)
            all_scores = torch.zeros(
                (
                    len(
                        all_track_ids,
                    )
                ),
                device=all_track_ids.device,
            )
            all_boxes = torch.zeros(
                (len(all_track_ids), 4), device=all_track_ids.device
            )
            all_embeddings = torch.zeros(
                (len(all_track_ids), memory_embeddings.size(1)),
                device=all_track_ids.device,
            )

            # calculate exponential moving average of embedding across memory
            for i, track_id in enumerate(all_track_ids):
                track_mask = (memory_track_ids == track_id).nonzero()[0]
                all_boxes[i] = memory_boxes[track_mask][0]
                all_scores[i] = memory_scores[track_mask][0]
                all_class_ids[i] = memory_class_ids[track_mask][0]
                embeddings = memory_embeddings[track_mask]
                embedding = embeddings[0]
                for mem_embed in embeddings[1:]:
                    embedding = (
                        1 - self.memo_momentum
                    ) * embedding + self.memo_momentum * mem_embed
                all_embeddings[i] = embedding
        else:
            all_track_ids = torch.empty((0,), dtype=torch.int64, device=device)
            all_class_ids = torch.empty((0,), dtype=torch.int64, device=device)
            all_scores = torch.empty((0,), device=device)
            all_boxes = torch.empty((0, 4), device=device)
            all_embeddings = torch.empty((0, 1), device=device)

        return QDTrackState(
            all_track_ids, all_boxes, all_scores, all_class_ids, all_embeddings
        )


# @torch.jit.script TODO
class QDTrackAssociation:
    """Tracking component relying on quasi-dense instance similarity.

    This class assigns detection candidates to a given memory of existing
    tracks.

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
        self.nms_conf_thr = nms_conf_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.with_cats = with_cats
        self.embed_dim = 256  # TODO

    def _filter_detections(
        self,
        detections: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Remove overlapping objects across classes via nms.

        Args:
            detections (torch.Tensor): _description_
            scores (torch.Tensor): _description_
            class_ids (torch.Tensor): _description_
            embeddings (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        # duplicate removal for potential backdrops and cross classes
        scores, inds = scores.sort(descending=True)
        detections, embeddings = detections[inds], embeddings[inds]
        valids = scores > self.obj_score_thr
        ious = bbox_iou(detections, detections)
        for i in range(1, len(detections)):
            if scores[i] < self.obj_score_thr:
                thr = self.nms_backdrop_iou_thr
            else:
                thr = self.nms_class_iou_thr

            if (ious[i, :i] > thr).any():
                valids[i] = False
        detections = detections[valids]
        scores = scores[valids]
        class_ids = class_ids[valids]
        embeddings = embeddings[valids]
        return detections, scores, class_ids, embeddings, inds[valids]

    def __call__(
        self,
        detections: torch.Tensor,
        detection_scores: torch.Tensor,
        detection_class_ids: torch.Tensor,
        detection_embeddings: torch.Tensor,
        memory_track_ids: torch.Tensor,
        memory_class_ids: torch.Tensor,
        memory_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process inputs, match detections with existing tracks."""
        (
            detections,
            detection_scores,
            detection_class_ids,
            detection_embeddings,
            permute_inds,
        ) = self._filter_detections(
            detections,
            detection_scores,
            detection_class_ids,
            detection_embeddings,
        )
        if len(detections) == 0:
            return torch.empty(
                (0,), dtype=torch.long, device=detections.device
            ), torch.empty((0,), dtype=torch.long, device=detections.device)

        # match if buffer is not empty
        if len(memory_track_ids) > 0:
            affinity_scores = calc_bisoftmax_affinity(
                detection_class_ids,
                detection_embeddings,
                memory_class_ids,
                memory_embeddings,
            )
            ids = greedy_assign(
                detection_scores, memory_track_ids, affinity_scores
            )
        else:
            ids = torch.full(
                (len(detections),),
                -1,
                dtype=torch.long,
                device=detections.device,
            )

        new_inds = (ids == -1) & (detection_scores > self.init_score_thr)
        ids[new_inds] = random_ids(new_inds.sum(), device=ids.device)
        return ids[permute_inds], permute_inds
