"""QD-3DT tracking graph."""
import copy
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import torch
from torch import nn
import torch.nn.functional as F

from vis4d.common.bbox.utils import bbox_iou
from vis4d.model.track.motion import (
    MotionModelConfig,
    build_lstm_model,
    build_motion_model,
)
from vis4d.struct import Boxes2D, Boxes3D, InputSample, LabelInstances

from .base import TrackGraphConfig
from .qdtrack import QDTrackGraph, QDTrackGraphConfig


class QD3DTrackGraphConfig(QDTrackGraphConfig):
    """QD-3DT track graph config."""

    motion_momentum: float = 0.8
    bbox_affinity_weight: float = 0.5
    motion_model: MotionModelConfig
    lstm_name: Optional[str] = None
    lstm_ckpt_name: Optional[str] = None
    feature_dim: int = 64
    hidden_size: int = 128
    num_layers: int = 2


class Track(TypedDict):
    """Track representation for QD3DT."""

    bbox: torch.Tensor
    bbox_3d: torch.Tensor
    motion_model: nn.Module
    embed: torch.Tensor
    class_id: torch.Tensor
    last_frame: int
    velocity: torch.Tensor
    acc_frame: int


class QD3DTrackGraph(QDTrackGraph):
    """Tracking graph construction for QD-3DT."""

    def __init__(self, cfg: TrackGraphConfig) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg = QD3DTrackGraphConfig(**cfg.dict())
        self.cfg.motion_dims = self.cfg.motion_model.motion_dims

        if self.cfg.lstm_name is not None:
            self.cfg.motion_model.lstm = build_lstm_model(self.cfg)

        self.cfg.feat_affinity_weight = 1 - self.cfg.bbox_affinity_weight

    def reset(self) -> None:
        """Reset tracks."""
        self.num_tracks = 0
        self.tracks: Dict[int, Track] = {}
        self.backdrops: List[
            Dict[str, Union[Boxes2D, Boxes3D, nn.Module, torch.Tensor]]
        ] = []

    def get_tracks(
        self,
        device: torch.device,
        frame_id: Optional[int] = None,
        add_backdrops: bool = False,
    ) -> Tuple[Boxes2D, Boxes3D, torch.Tensor]:
        """Get active tracks at given frame.

        If frame_id is None, return all tracks in memory.
        """
        (
            bboxs,
            bboxs_3d,
            embeds,
            motion_models,
            velocities,
            class_ids,
            ids,
        ) = ([], [], [], [], [], [], [])
        for k, v in self.tracks.items():
            if frame_id is None or v["last_frame"] == frame_id:
                bboxs.append(v["bbox"].unsqueeze(0))
                bboxs_3d.append(v["bbox_3d"].unsqueeze(0))
                embeds.append(v["embed"].unsqueeze(0))
                motion_models.append(v["motion_model"])
                velocities.append(v["velocity"].unsqueeze(0))
                class_ids.append(v["class_id"])
                ids.append(k)

        bboxs = (
            torch.cat(bboxs)
            if len(bboxs) > 0
            else torch.empty((0, 5), device=device)
        )
        bboxs_3d = (
            torch.cat(bboxs_3d)
            if len(bboxs_3d) > 0
            else torch.empty((0, 10), device=device)
        )
        embeds = (
            torch.cat(embeds)
            if len(embeds) > 0
            else torch.empty((0,), device=device)
        )
        velocities = (
            torch.cat(velocities)
            if len(velocities) > 0
            else torch.empty((0, self.cfg.motion_dims), device=device)
        )
        class_ids = (
            torch.cat(class_ids)
            if len(class_ids) > 0
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
                bboxs_3d = torch.cat(
                    [bboxs_3d, backdrop["detections_3d"].boxes]
                )
                embeds = torch.cat([embeds, backdrop["embeddings"]])
                motion_models.extend(backdrop["motion_model"])
                backdrop_vs = torch.zeros_like(
                    backdrop["detections_3d"].boxes[:, : self.cfg.motion_dims]
                )
                velocities = torch.cat([velocities, backdrop_vs])
                class_ids = torch.cat(
                    [class_ids, backdrop["detections"].class_ids]
                )

        return (
            Boxes2D(bboxs, class_ids, ids),
            Boxes3D(bboxs_3d, class_ids, ids),
            embeds,
            motion_models,
            velocities,
        )

    def remove_duplicates(
        self,
        detections: Boxes2D,
        detections_3d: Boxes3D,
        embeddings: torch.Tensor,
    ) -> Tuple[Boxes2D, torch.Tensor]:
        """Remove overlapping objects across classes via nms."""
        # duplicate removal for potential backdrops and cross classes
        scores = detections.score
        assert scores is not None
        scores, inds = scores.sort(descending=True)
        detections, detections_3d, embeddings = (
            detections[inds],
            detections_3d[inds],
            embeddings[inds],
        )
        valids = embeddings.new_ones((len(detections),))
        ious = bbox_iou(detections, detections)
        for i in range(1, len(detections)):
            if scores[i] < self.cfg.obj_score_thr:
                thr = self.cfg.nms_backdrop_iou_thr
            else:
                thr = self.cfg.nms_class_iou_thr

            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        detections = detections[valids, :]
        detections_3d = detections_3d[valids, :]
        embeddings = embeddings[valids, :]
        return detections, detections_3d, embeddings, valids, inds

    def depth_ordering(
        self,
        obsv_boxes_3d,
        memo_boxes_3d_predict,
        memo_boxes_3d,
        memo_vs,
    ):
        centroid_weight = F.pairwise_distance(
            obsv_boxes_3d[..., :3, None],
            memo_boxes_3d_predict[..., :3, None].transpose(2, 0),
        )
        centroid_weight = torch.exp(-centroid_weight / 10.0)
        # Moving distance should be aligned
        # V_observed-tracked vs. V_velocity
        motion_weight = F.pairwise_distance(
            obsv_boxes_3d[..., :3, None]
            - memo_boxes_3d[..., :3, None].transpose(2, 0),
            memo_vs[..., :3, None].transpose(2, 0),
        )
        motion_weight = torch.exp(-motion_weight / 5.0)
        # Moving direction should be aligned
        # Set to 0.5 when two vector not within +-90 degree
        cos_sim = F.cosine_similarity(
            obsv_boxes_3d[..., :2, None]
            - memo_boxes_3d[..., :2, None].transpose(2, 0),
            memo_vs[..., :2, None].transpose(2, 0),
        )
        cos_sim += 1.0
        cos_sim /= 2.0
        scores_depth = (
            cos_sim * centroid_weight + (1.0 - cos_sim) * motion_weight
        )

        return scores_depth

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
        ), "QD3DTrackGraph requires instance embeddings."
        assert len(inputs) == 1, "QD3DTrackGraph support only BS=1 inference."
        detections = predictions.boxes2d[0].clone()
        detections_3d = predictions.boxes3d[0].clone()
        embeddings = embeddings[0].clone()
        frame_id = inputs.metadata[0].frameIndex
        assert (
            frame_id is not None
        ), "Couldn't find current frame index in InputSample metadata!"

        # reset graph at begin of sequence
        if frame_id == 0:
            self.reset()

        (
            detections,
            detections_3d,
            embeddings,
            valids,
            permute_inds,
        ) = self.remove_duplicates(detections, detections_3d, embeddings)

        # init ids container
        ids = torch.full(
            (len(detections),), -1, dtype=torch.long, device=detections.device
        )

        # match if buffer is not empty
        detections_scores = detections.score
        depth_confidence = detections_3d.score
        assert detections_scores is not None

        if len(detections) > 0 and not self.empty:
            (
                memo_dets,
                memo_dets_3d,
                memo_embeds,
                memo_motion_models,
                memo_vs,
            ) = self.get_tracks(detections.device, add_backdrops=True)

            memo_obs_3d = self.parse_observation(
                memo_dets_3d.boxes, batch=True
            )

            memo_boxes_3d_predict = memo_obs_3d.clone()
            for ind, memo_motion_model in enumerate(memo_motion_models):
                memo_velo = memo_motion_model.predict(
                    update_state=memo_motion_model.cfg.age != 0
                )[self.cfg.motion_dims :]
                memo_boxes_3d_predict[ind, :3] += memo_velo

            obs_3d = self.parse_observation(detections_3d.boxes, batch=True)

            # BBox IoU
            depth_weight = F.pairwise_distance(
                obs_3d[:, : self.cfg.motion_dims][..., None],
                memo_boxes_3d_predict[:, : self.cfg.motion_dims][
                    ..., None
                ].transpose(2, 0),
            )
            scores_iou = torch.exp(-depth_weight / 10.0)

            # Depth Ordering
            scores_depth = self.depth_ordering(
                obs_3d[:, : self.cfg.motion_dims],
                memo_obs_3d[:, : self.cfg.motion_dims],
                memo_boxes_3d_predict[:, : self.cfg.motion_dims],
                memo_vs[:, : self.cfg.motion_dims],
            )

            # Quasi Dense
            feats = torch.mm(embeddings, memo_embeds.t())
            d2t_scores = feats.softmax(dim=1)
            t2d_scores = feats.softmax(dim=0)
            similarity_scores = (d2t_scores + t2d_scores) / 2

            # Score with categories
            if self.cfg.with_cats:
                scores_cats = detections.class_ids.view(
                    -1, 1
                ) == memo_dets.class_ids.view(1, -1)

            scores = (
                self.cfg.bbox_affinity_weight * scores_iou * scores_depth
                + self.cfg.feat_affinity_weight * similarity_scores
            )
            scores /= (
                self.cfg.bbox_affinity_weight + self.cfg.feat_affinity_weight
            )
            scores *= (scores_iou > 0.0).float()
            scores *= (scores_depth > 0.0).float()
            scores *= scores_cats

            for i in range(len(detections)):
                conf, memo_ind = torch.max(scores[i, :], dim=0)
                cur_id = memo_dets.track_ids[memo_ind]
                if conf > self.cfg.match_score_thr:
                    if cur_id > -1:
                        if (
                            detections_scores[i] * depth_confidence[i]
                            > self.cfg.obj_score_thr
                        ):
                            ids[i] = cur_id
                            scores[:i, memo_ind] = 0
                            scores[(i + 1) :, memo_ind] = 0
                        elif conf > self.cfg.nms_conf_thr:  # pragma: no cover
                            ids[i] = -2

        new_inds = (ids == -1) & (detections_scores > self.cfg.init_score_thr)
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracks,
            self.num_tracks + num_news,
            dtype=torch.long,
            device=ids.device,
        )
        self.num_tracks += num_news

        self.update(ids, detections, detections_3d, embeddings, frame_id)

        # remove backdrops, low score
        valids[valids.clone()] = ids > -1
        result = copy.deepcopy(predictions)
        for pred in result.get_instance_labels():
            if len(pred[0]) > 0:  # type: ignore
                pred[0] = pred[0][permute_inds][valids]  # type: ignore
                pred[0].track_ids = ids[ids > -1]  # type: ignore
                if isinstance(pred[0], Boxes3D):
                    for i, tid in enumerate(ids[ids > -1]):
                        pred[0].boxes[i] = self.tracks[int(tid)]["bbox_3d"]

        return result

    def update(  # type: ignore # pylint: disable=arguments-differ
        self,
        ids: torch.Tensor,
        detections: Boxes2D,
        detections_3d: Boxes3D,
        embeddings: torch.Tensor,
        frame_id: int,
    ) -> None:
        """Update track memory using matched detections."""
        tracklet_inds = ids > -1

        # update memo
        for cur_id, det, det3d, embed in zip(
            ids[tracklet_inds],
            detections[tracklet_inds],
            detections_3d[tracklet_inds],
            embeddings[tracklet_inds],
        ):  # type: ignore
            cur_id = int(cur_id)
            if cur_id in self.tracks.keys():
                self.update_track(cur_id, det, det3d, embed, frame_id)
            else:
                self.create_track(cur_id, det, det3d, embed, frame_id)

        # Handle vanished tracklets
        for track_id in self.tracks:
            if (
                frame_id > self.tracks[track_id]["last_frame"]
                and track_id > -1
            ):
                pd_box_3d = self.tracks[track_id]["motion_model"].predict()
                self.tracks[track_id]["bbox_3d"][:6] = pd_box_3d[:6]
                self.tracks[track_id]["bbox_3d"][7] = pd_box_3d[6]

        backdrop_inds = torch.nonzero(ids == -1, as_tuple=False).squeeze(1)
        ious = bbox_iou(detections[backdrop_inds], detections)
        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.cfg.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]

        backdrop_motion_model = [
            build_motion_model(
                self.cfg.motion_model,
                detections_3d[bd_ind].boxes[0],
            )
            for bd_ind in backdrop_inds
        ]

        self.backdrops.insert(
            0,
            dict(
                detections=detections[backdrop_inds],
                detections_3d=detections_3d[backdrop_inds],
                embeddings=embeddings[backdrop_inds],
                motion_model=backdrop_motion_model,
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
        detection_3d: Boxes3D,
        embedding: torch.Tensor,
        frame_id: int,
    ) -> None:
        """Update a specific track with a new models."""
        bbox = detection.boxes[0]
        class_id = detection.class_ids[0]
        bbox_3d = detection_3d.boxes[0]
        obs_3d = self.parse_observation(bbox_3d)
        self.tracks[track_id]["bbox"] = bbox
        self.tracks[track_id]["motion_model"].update(obs_3d)

        pd_box_3d = self.tracks[track_id]["motion_model"].get_state()[
            : self.cfg.motion_dims
        ]

        prev_obs = self.parse_observation(self.tracks[track_id]["bbox_3d"])
        velocity = (pd_box_3d - prev_obs[: self.cfg.motion_dims]) / (
            frame_id - self.tracks[track_id]["last_frame"]
        )

        # Update Box3D with center, dim, rot_y
        self.tracks[track_id]["bbox_3d"][:6] = pd_box_3d[:6]
        self.tracks[track_id]["bbox_3d"][7] = pd_box_3d[6]

        self.tracks[track_id]["embed"] = (
            1 - self.cfg.memo_momentum
        ) * self.tracks[track_id]["embed"] + self.cfg.memo_momentum * embedding
        self.tracks[track_id]["last_frame"] = frame_id
        self.tracks[track_id]["class_id"] = class_id
        self.tracks[track_id]["velocity"] = (
            self.tracks[track_id]["velocity"]
            * self.tracks[track_id]["acc_frame"]
            + velocity
        ) / (self.tracks[track_id]["acc_frame"] + 1)
        self.tracks[track_id]["acc_frame"] += 1

    def parse_observation(
        self, bbox_3d: torch.Tensor, batch: bool = False
    ) -> torch.Tensor:
        """Parse required boudning box."""
        if batch:
            obs_3d = torch.zeros(
                (bbox_3d.shape[0], self.cfg.motion_dims + 1)
            ).to(bbox_3d.device)
            obs_3d[:, :6] = bbox_3d[:, :6]
            obs_3d[:, 6] = bbox_3d[:, 7]
            obs_3d[:, 7] = bbox_3d[:, -1]
        else:
            obs_3d = torch.zeros(self.cfg.motion_dims + 1).to(bbox_3d.device)
            obs_3d[:6] = bbox_3d[:6]
            obs_3d[6] = bbox_3d[7]
            obs_3d[7] = bbox_3d[-1]
        return obs_3d

    def create_track(
        self,
        track_id: int,
        detection: Boxes2D,
        detection_3d: Boxes3D,
        embedding: torch.Tensor,
        frame_id: int,
    ) -> None:
        """Create a new track from a models."""
        bbox = detection.boxes[0]
        class_id = detection.class_ids[0]
        bbox_3d = detection_3d.boxes[0]
        obs_3d = self.parse_observation(bbox_3d)
        motion_model = build_motion_model(self.cfg.motion_model, obs_3d)
        self.tracks[track_id] = dict(
            bbox=bbox,
            bbox_3d=bbox_3d,
            motion_model=motion_model,
            embed=embedding,
            class_id=class_id,
            last_frame=frame_id,
            velocity=torch.zeros(self.cfg.motion_dims).to(bbox_3d.device),
            acc_frame=0,
        )
