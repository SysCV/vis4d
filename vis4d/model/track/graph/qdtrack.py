"""Quasi-dense embedding similarity based graph."""
import copy
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import torch

from vis4d.common.bbox.utils import bbox_iou
from vis4d.struct import Boxes2D, InputSample, LabelInstances, LossesType

from .base import BaseTrackGraph


class Track(TypedDict):
    """Track representation for QDTrack."""

    bbox: torch.Tensor
    embed: torch.Tensor
    class_id: torch.Tensor
    last_frame: int
    velocity: torch.Tensor
    acc_frame: int


class QDTrackGraph(BaseTrackGraph):
    """Tracking graph construction for quasi-dense instance similarity."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        keep_in_memory: int,  # threshold for keeping occluded objects in mem
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

        # validate arguments
        assert 0 <= memo_momentum <= 1.0
        assert keep_in_memory >= 0
        assert memo_backdrop_frames >= 0

        self.reset()

    def reset(self) -> None:
        """Reset tracks."""
        self.num_tracks = 0
        self.tracks: Dict[int, Track] = {}
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
        ids = torch.tensor(ids, device=device)

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
    ) -> Tuple[Boxes2D, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove overlapping objects across classes via nms."""
        # duplicate removal for potential backdrops and cross classes
        scores = detections.score
        assert scores is not None
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

    def forward_train(
        self,
        inputs: List[InputSample],
        predictions: List[LabelInstances],
        targets: Optional[List[LabelInstances]],
        **kwargs: List[torch.Tensor],
    ) -> LossesType:
        """Forward of QDTrackGraph in training stage."""
        raise NotImplementedError

    def forward_test(
        self,
        inputs: InputSample,
        predictions: LabelInstances,
        embeddings: Optional[torch.Tensor] = None,
        **kwargs: torch.Tensor,
    ) -> LabelInstances:
        """Process inputs, match detections with existing tracks."""
        assert (
            embeddings is not None
        ), "QDTrackGraph requires instance embeddings."
        assert len(inputs) == 1, "QDTrackGraph support only BS=1 inference."
        detections = predictions.boxes2d[0].clone()
        embeddings = embeddings[0].clone()
        frame_id = inputs.metadata[0].frameIndex
        assert (
            frame_id is not None
        ), "Couldn't find current frame index in InputSample metadata!"

        # reset graph at begin of sequence
        if frame_id == 0:
            self.reset()

        detections, embeddings, valids, permute_inds = self.remove_duplicates(
            detections, embeddings
        )

        # init ids container
        ids = torch.full(
            (len(detections),), -1, dtype=torch.long, device=detections.device
        )

        # match if buffer is not empty
        detections_scores = detections.score
        assert detections_scores is not None
        if len(detections) > 0 and not self.empty:
            memo_dets, memo_embeds = self.get_tracks(
                detections.device, add_backdrops=True
            )

            # match using bisoftmax metric
            feats = torch.mm(embeddings, memo_embeds.t())
            d2t_scores = feats.softmax(dim=1)
            t2d_scores = feats.softmax(dim=0)
            similarity_scores = (d2t_scores + t2d_scores) / 2

            if self.with_cats:
                cat_same = detections.class_ids.view(
                    -1, 1
                ) == memo_dets.class_ids.view(1, -1)
                similarity_scores *= cat_same.float()

            for i in range(len(detections)):
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
            if frame_id - v["last_frame"] >= self.keep_in_memory:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
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
            1 - self.memo_momentum
        ) * self.tracks[track_id]["embed"] + self.memo_momentum * embedding
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
        self.tracks[track_id] = Track(
            bbox=bbox,
            embed=embedding,
            class_id=cls,
            last_frame=frame_id,
            velocity=torch.zeros_like(bbox),
            acc_frame=0,
        )
