"""QD-3DT tracking graph."""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from vist.common.bbox.utils import bbox_iou, get_yaw_cam, get_yaw_world
from vist.model.track.motion import (
    LSTM3DMotionModelConfig,
    VeloLSTM,
    build_motion_model,
)
from vist.struct import Boxes2D, Boxes3D

from .base import TrackGraphConfig
from .quasi_dense import QDTrackGraph, QDTrackGraphConfig


class QD3DTrackGraphConfig(QDTrackGraphConfig):
    """Quasi-dense similarity based graph config."""

    motion_momentum: float
    bbox_affinity_weight: float
    motion_model: LSTM3DMotionModelConfig


class QD3DTrackGraph(QDTrackGraph):
    """Tracking graph construction for quasi-dense instance similarity."""

    def __init__(self, cfg: TrackGraphConfig) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg = QD3DTrackGraphConfig(**cfg.dict())
        self.cfg.motion_dims = self.cfg.motion_model.motion_dims

        if self.cfg.motion_model.type == "LSTM3DMotionModel":
            self.cfg.motion_model.lstm = VeloLSTM(
                1, 64, 128, 2, self.cfg.motion_model.motion_dims
            )
            if self.cfg.motion_model.lstm_ckpt_name is not None:
                ckpt = torch.load(self.cfg.motion_model.lstm_ckpt_name)
                try:
                    self.cfg.motion_model.lstm.load_state_dict(
                        ckpt["state_dict"]
                    )
                except (RuntimeError, KeyError) as ke:
                    print("Cannot load full model: {}".format(ke))
                    state = self.cfg.motion_model.lstm.state_dict()
                    state.update(ckpt["state_dict"])
                    self.cfg.motion_model.lstm.load_state_dict(state)
                del ckpt

        self.cfg.feat_affinity_weight = 1 - self.cfg.bbox_affinity_weight

    def reset(self) -> None:
        """Reset tracks."""
        super().reset()
        self.backdrops = []

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
            boxes_3d,
            embeds,
            motion_models,
            velocities,
            class_ids,
            ids,
        ) = ([], [], [], [], [], [], [])
        for k, v in self.tracks.items():
            if frame_id is None or v["last_frame"] == frame_id:
                bboxs.append(v["bbox"].unsqueeze(0))
                boxes_3d.append(v["box_3d"].unsqueeze(0))
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
        boxes_3d = (
            torch.cat(boxes_3d)
            if len(boxes_3d) > 0
            else torch.empty((0, self.cfg.motion_dims + 1), device=device)
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
                boxes_3d = torch.cat(
                    [boxes_3d, backdrop["detections_3d"].boxes]
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
            Boxes3D(boxes_3d, class_ids, ids),
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
        _, inds = detections.boxes[:, -1].sort(descending=True)
        detections, embeddings = detections[inds], embeddings[inds]
        valids = embeddings.new_ones((len(detections),))
        ious = bbox_iou(detections, detections)
        for i in range(1, len(detections)):
            if detections.boxes[i, -1] < self.cfg.obj_score_thr:
                thr = self.cfg.nms_backdrop_iou_thr
            else:
                thr = self.cfg.nms_class_iou_thr

            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        detections = detections[valids, :]
        detections_3d = detections_3d[valids, :]
        embeddings = embeddings[valids, :]
        return detections, detections_3d, embeddings

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

    def match(self, detections, detections_3d, embeddings, ids):
        (
            memo_dets,
            memo_dets_3d,
            memo_embeds,
            memo_motion_models,
            memo_vs,
        ) = self.get_tracks(detections.device, add_backdrops=True)

        memo_boxes_3d_predict = memo_dets_3d.boxes.detach().clone()
        for ind, memo_motion_model in enumerate(memo_motion_models):
            memo_velo = memo_motion_model.predict(
                update_state=memo_motion_model.cfg.age != 0
            )
            memo_boxes_3d_predict[ind, :3] += memo_velo[7:]

        # BBox IoU
        depth_weight = F.pairwise_distance(
            detections_3d.boxes[:, :7][..., None],
            memo_boxes_3d_predict[:, :7][..., None].transpose(2, 0),
        )
        scores_iou = torch.exp(-depth_weight / 10.0)

        # Quasi Dense
        feats = torch.mm(embeddings, memo_embeds.t())
        d2t_scores = feats.softmax(dim=1)
        t2d_scores = feats.softmax(dim=0)
        scores_embedding = (d2t_scores + t2d_scores) / 2

        # Depth Ordering
        scores_depth = self.depth_ordering(
            detections_3d.boxes[:, :7],
            memo_dets_3d.boxes[:, :7],
            memo_boxes_3d_predict[:, :7],
            memo_vs[:, :7],
        )

        if self.cfg.with_cats:
            cat_same = detections.class_ids.view(
                -1, 1
            ) == memo_dets.class_ids.view(1, -1)
            scores_cats = cat_same.float()

        scores = (
            self.cfg.bbox_affinity_weight * scores_iou * scores_depth
            + self.cfg.feat_affinity_weight * scores_embedding
        )
        scores /= self.cfg.bbox_affinity_weight + self.cfg.feat_affinity_weight
        scores *= (scores_iou > 0.0).float()
        scores *= (scores_depth > 0.0).float()
        scores *= scores_cats

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

        return ids

    def forward(  # type: ignore # pylint: disable=arguments-differ
        self,
        detections: Boxes2D,
        detections_3d: Boxes3D,
        frame_id: int,
        embeddings: torch.Tensor,
        cam_extrinsics: torch.Tensor,
    ) -> Tuple[Boxes2D, Boxes3D]:
        """Process inputs, match detections with existing tracks."""
        detections, detections_3d, embeddings = self.remove_duplicates(
            detections, detections_3d, embeddings
        )

        quat_det_yaws_world = get_yaw_world(
            detections_3d.boxes[:, 6],
            cam_extrinsics.detach().cpu().numpy(),
        )

        detections_3d.boxes[:, 6] = torch.from_numpy(
            quat_det_yaws_world["yaw_world"]
        )

        # init ids container
        ids = torch.full(
            (len(detections),), -1, dtype=torch.long, device=detections.device
        )

        # match if buffer is not empty
        if len(detections) > 0 and not self.empty:
            ids = self.match(detections, detections_3d, embeddings, ids)

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

        self.update(ids, detections, detections_3d, embeddings, frame_id)
        result, result_3d, _, _, _ = self.get_tracks(
            detections.device, frame_id
        )

        yaws_cam = get_yaw_cam(
            result_3d.boxes[:, 6],
            cam_extrinsics.detach().cpu().numpy(),
            quat_det_yaws_world,
        )

        result_3d.boxes[:, 6] = torch.from_numpy(yaws_cam)

        return result, result_3d

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
                self.tracks[track_id]["box_3d"][
                    : self.cfg.motion_dims
                ] = self.tracks[track_id]["motion_model"].predict()[
                    : self.cfg.motion_dims
                ]

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
        bbox, det_3d, class_id = (
            detection.boxes[0],
            detection_3d.boxes[0],
            detection.class_ids[0],
        )
        self.tracks[track_id]["bbox"] = bbox
        self.tracks[track_id]["motion_model"].update(det_3d)

        pd_box_3d = self.tracks[track_id]["motion_model"].get_state()[
            : self.cfg.motion_dims
        ]

        velocity = (
            pd_box_3d - self.tracks[track_id]["box_3d"][: self.cfg.motion_dims]
        ) / (frame_id - self.tracks[track_id]["last_frame"])

        self.tracks[track_id]["box_3d"][: self.cfg.motion_dims] = pd_box_3d
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

    def create_track(
        self,
        track_id: int,
        detection: Boxes2D,
        detection_3d: Boxes3D,
        embedding: torch.Tensor,
        frame_id: int,
    ) -> None:
        """Create a new track from a models."""
        bbox, det_3d, class_id = (
            detection.boxes[0],
            detection_3d.boxes[0],
            detection.class_ids[0],
        )
        motion_model = build_motion_model(
            self.cfg.motion_model,
            det_3d,
        )
        self.tracks[track_id] = dict(
            bbox=bbox,
            box_3d=det_3d,
            motion_model=motion_model,
            embed=embedding,
            class_id=class_id,
            last_frame=frame_id,
            velocity=torch.zeros_like(det_3d[: self.cfg.motion_dims]),
            acc_frame=0,
        )
