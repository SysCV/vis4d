"""CC-3DT model implementation.

This file composes the operations associated with
CC-3DT `https://arxiv.org/abs/2212.01247' into the full model implementation.
"""
from __future__ import annotations
from typing import NamedTuple

import torch
from torch import Tensor, nn

from vis4d.common import ArgsType
from vis4d.engine.ckpt import load_model_checkpoint
from vis4d.model.track.qdtrack import QDTrack
from vis4d.op.base import ResNet
from vis4d.op.detect.faster_rcnn import (
    AnchorGenerator,
    FasterRCNNHead,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.op.detect.rcnn import RoI2Det
from vis4d.op.detect_3d.filter import bev_3d_nms
from vis4d.op.detect_3d.qd_3dt import QD3DTBBox3DHead
from vis4d.op.fpp import FPN
from vis4d.op.track.assignment import TrackIDCounter
from vis4d.op.track_3d.cc_3dt import CC3DTrackAssociation, cam_to_global
from vis4d.op.track.motion.kalman_filter import predict
from vis4d.op.track_3d.motion.kf3d import (
    kf3d_init,
    kf3d_init_mean_cov,
    kf3d_predict,
    kf3d_update,
)
from vis4d.op.track_3d.motion.lstm_3d import VeloLSTM
from vis4d.state.track.cc_3dt import CC3DTrackMemory, CC3DTrackState
from vis4d.op.detect.rcnn import RCNNHead


class Track3DOut(NamedTuple):
    """Output of track 3D model."""

    boxes_3d: Tensor  # (N, 12): x,y,z,h,w,l,rx,ry,rz,vx,vy,vz
    class_ids: Tensor
    scores_3d: Tensor
    track_ids: Tensor


def get_default_anchor_generator() -> AnchorGenerator:
    """Get default anchor generator."""
    return AnchorGenerator(
        scales=[4, 8],
        ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
        strides=[4, 8, 16, 32, 64],
    )


def get_default_roi_head(num_classes: int) -> RCNNHead:
    """Get default ROI head."""
    return RCNNHead(num_shared_convs=4, num_classes=num_classes)


class CC3DTrack(QDTrack):
    """CC-3DT model."""

    def __init__(
        self,
        *args: ArgsType,
        memory_size: int = 10,
        motion_model: str = "KF3D",
        motion_dims: int = 7,
        num_frames: int = 5,
        pure_det: bool = False,
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__(*args, memory_size=memory_size, **kwargs)
        self.track_memory = CC3DTrackMemory(
            memory_limit=memory_size,
            motion_dims=motion_dims,
            num_frames=num_frames,
        )
        self.track_graph = CC3DTrackAssociation()
        self.motion_model = motion_model
        self.motion_dims = motion_dims
        self.num_frames = num_frames
        self.pure_det = pure_det

        if self.motion_model == "KF3D":
            motion_mat, update_mat, cov_motion_Q, cov_project_R = kf3d_init(
                self.motion_dims
            )
            self.register_buffer("_motion_mat", motion_mat, False)  # F
            self.register_buffer("_update_mat", update_mat, False)  # H
            self.register_buffer("_cov_motion_Q", cov_motion_Q, False)  # Q
            self.register_buffer("_cov_project_R", cov_project_R, False)  # R
        else:
            self.lstm_model = VeloLSTM(
                feature_dim=64,
                hidden_size=128,
                num_layers=2,
                loc_dim=self.motion_dims,
            )
            self.num_frames = num_frames

    def _update_memory(
        self,
        frame_id: int,
        track_id: Tensor,
        update_attr: str,
        update_value: Tensor,
    ) -> None:
        """Update track memory."""
        track_indice = (
            self.track_memory.frames[frame_id].track_ids == track_id
        ).nonzero(as_tuple=False)[-1]
        frame = self.track_memory.frames[frame_id]
        state_value = list(getattr(frame, update_attr))
        state_value[track_indice] = update_value
        self.track_memory.replace_frame(
            frame_id, update_attr, torch.stack(state_value)
        )

    def _update_track(
        self,
        frame_id: Tensor,
        track_ids: Tensor,
        match_ids: Tensor,
        boxes_2d: Tensor,
        camera_ids: Tensor,
        scores_2d: Tensor,
        boxes_3d: Tensor,
        scores_3d: Tensor,
        class_ids: Tensor,
        embeddings: Tensor,
        obs_boxes_3d: Tensor,
        fps: int,
    ) -> CC3DTrackState:
        """Update track."""
        motion_states = []
        motion_hidden = []
        vel_histories = []
        velocities = []
        last_frames = []
        acc_frames = []
        for i, track_id in enumerate(track_ids):
            bbox_3d = boxes_3d[i]
            obs_3d = obs_boxes_3d[i]
            if track_id in match_ids:
                # update track
                tracks, _ = self.track_memory.get_track(track_id)
                track = tracks[-1]

                mean, covariance = kf3d_update(
                    self._update_mat.to(obs_3d.device),
                    self._cov_project_R.to(obs_3d.device),
                    track.motion_states[0],
                    track.motion_hidden[0],
                    obs_3d,
                )

                pd_box_3d = mean[: self.motion_dims]

                boxes_3d[i][:6] = pd_box_3d[:6]
                boxes_3d[i][8] = pd_box_3d[6]

                pred_loc, _ = predict(
                    self._motion_mat.to(obs_3d.device),
                    self._cov_motion_Q.to(obs_3d.device),
                    mean,
                    covariance,
                )
                boxes_3d[i][9:12] = (pred_loc[:3] - mean[:3]) * fps
                prev_obs = torch.cat(
                    [track.boxes_3d[0, :6], track.boxes_3d[0, 8].unsqueeze(0)]
                )
                velocity = (pd_box_3d - prev_obs) / (
                    frame_id - track.last_frames[0]
                )
                velocities.append(
                    (track.velocities[0] * track.acc_frames[0] + velocity)
                    / (track.acc_frames[0] + 1)
                )
                acc_frames.append(track.acc_frames[0] + 1)

                embeddings[i] = (
                    1 - self.memo_momentum
                ) * track.embeddings + self.memo_momentum * embeddings[i]

                motion_states.append(mean)
                motion_hidden.append(covariance)
                vel_histories.append(
                    torch.zeros(self.num_frames, self.motion_dims).to(
                        obs_3d.device
                    )
                )
            else:
                # create track
                if self.motion_model == "KF3D":
                    mean, covariance = kf3d_init_mean_cov(
                        obs_3d, self.motion_dims
                    )
                    motion_states.append(mean)
                    motion_hidden.append(covariance)
                elif self.motion_model == "VeloLSTM":
                    ref_hidden = self.lstm_model.init_hidden(obs_3d.device)
                    pred_hidden = self.lstm_model.init_hidden(obs_3d.device)
                    motion_states.append(ref_hidden)
                    motion_hidden.append(pred_hidden)
                else:
                    raise NotImplementedError
                vel_histories.append(
                    torch.zeros(self.num_frames, self.motion_dims).to(
                        obs_3d.device
                    )
                )
                velocities.append(
                    torch.zeros(self.motion_dims, device=bbox_3d.device)
                )
                acc_frames.append(0)
            last_frames.append(frame_id)

        motion_states = torch.stack(motion_states)
        motion_hidden = torch.stack(motion_hidden)
        velocities = torch.stack(velocities)
        vel_histories = torch.stack(vel_histories)
        last_frames = torch.tensor(last_frames, device=boxes_2d.device)
        acc_frames = torch.tensor(acc_frames, device=boxes_2d.device)

        return CC3DTrackState(
            track_ids,
            boxes_2d,
            camera_ids,
            scores_2d,
            boxes_3d,
            scores_3d,
            class_ids,
            embeddings,
            motion_states,
            motion_hidden,
            vel_histories,
            velocities,
            last_frames,
            acc_frames,
        )

    def _motion_predict(
        self,
        cur_memory: CC3DTrackState,
        index: int,
        track_id: int,
        device: torch.device,
        update: bool = True,
    ) -> Tensor:
        """Motion prediction."""
        if self.motion_model == "KF3D":
            pd_box_3d, cov = kf3d_predict(
                self._motion_mat.to(device),
                self._cov_motion_Q.to(device),
                cur_memory.motion_states[index],
                cur_memory.motion_hidden[index],
            )
            if update:
                _, fids = self.track_memory.get_track(track_id)

                if len(fids) > 0:
                    self._update_memory(
                        fids[-1], track_id, "motion_states", pd_box_3d
                    )
                    self._update_memory(
                        fids[-1], track_id, "motion_hidden", cov
                    )
        elif self.motion_model == "VeloLSTM":
            with torch.no_grad():
                pd_box_3d, hidden_pred = self.lstm_model.predcit(
                    cur_memory.vel_history[index],
                    cur_memory.motion_hidden[index],
                )
        else:
            raise NotImplementedError

        return pd_box_3d

    def _forward_test(
        self,
        features_list: list[torch.Tensor],
        boxes_2d_list: list[torch.Tensor],
        scores_2d_list: list[torch.Tensor],
        boxes_3d_list: list[torch.Tensor],
        scores_3d_list: list[torch.Tensor],
        class_ids_list: list[torch.Tensor],
        frame_ids: list[int],
        extrinsics: torch.Tensor,
        class_range_map: None | Tensor = None,
        fps: int = 2,
    ) -> Track3DOut:
        """Forward function during testing."""
        embeddings_list = list(
            self.similarity_head(features_list, boxes_2d_list)
        )

        (
            boxes_2d,
            camera_ids,
            scores_2d,
            boxes_3d,
            scores_3d,
            class_ids,
            embeddings,
        ) = cam_to_global(
            boxes_2d_list,
            scores_2d_list,
            boxes_3d_list,
            scores_3d_list,
            class_ids_list,
            embeddings_list,
            extrinsics,
            class_range_map,
        )

        if self.pure_det:
            empty_track = self.track_memory.get_empty_frame(
                len(class_ids), boxes_2d.device
            )
            return CC3DTrackState(
                empty_track.track_ids,
                boxes_2d,
                camera_ids,
                scores_2d,
                boxes_3d,
                scores_3d,
                class_ids,
                embeddings,
                empty_track.motion_states,
                empty_track.motion_hidden,
                empty_track.vel_histories,
                empty_track.velocities,
                empty_track.last_frames,
                empty_track.acc_frames,
            )

        # merge multi-view boxes
        keep_indices = bev_3d_nms(
            boxes_3d,
            scores_2d * scores_3d,
            class_ids,
        )

        boxes_2d = boxes_2d[keep_indices]
        camera_ids = camera_ids[keep_indices]
        scores_2d = scores_2d[keep_indices]
        boxes_3d = boxes_3d[keep_indices]
        scores_3d = scores_3d[keep_indices]
        class_ids = class_ids[keep_indices]
        embeddings = embeddings[keep_indices]

        for frame_id in frame_ids:
            assert (
                frame_id == frame_ids[0]
            ), "All cameras should have same frame_id."
        frame_id = frame_ids[0]
        tracks = []

        # reset graph at begin of sequence
        if frame_id == 0:
            self.track_memory.reset()
            TrackIDCounter.reset()

        cur_memory = self.track_memory.get_current_tracks(boxes_2d.device)

        memory_boxes_3d = torch.cat(
            [
                cur_memory.boxes_3d[:, :6],
                cur_memory.boxes_3d[:, 8].unsqueeze(1),
            ],
            dim=1,
        )

        if len(cur_memory.track_ids) > 0:
            memory_boxes_3d_predict = memory_boxes_3d.clone()
            for i, track_id in enumerate(cur_memory.track_ids):
                pd_box_3d = self._motion_predict(
                    cur_memory, i, track_id, boxes_2d.device
                )
                memory_boxes_3d_predict[i, :3] += pd_box_3d[self.motion_dims :]
        else:
            memory_boxes_3d_predict = torch.empty(
                (0, 7), device=boxes_2d.device
            )

        obs_boxes_3d = torch.cat(
            [boxes_3d[:, :6], boxes_3d[:, 8].unsqueeze(1)], dim=1
        )

        track_ids, match_ids, filter_indices = self.track_graph(
            boxes_2d,
            camera_ids,
            scores_2d,
            obs_boxes_3d,
            scores_3d,
            class_ids,
            embeddings,
            memory_boxes_3d,
            cur_memory.track_ids,
            cur_memory.class_ids,
            cur_memory.embeddings,
            memory_boxes_3d_predict,
            cur_memory.velocities,
        )

        data = self._update_track(
            frame_id,
            track_ids,
            match_ids,
            boxes_2d[filter_indices],
            camera_ids[filter_indices],
            scores_2d[filter_indices],
            boxes_3d[filter_indices],
            scores_3d[filter_indices],
            class_ids[filter_indices],
            embeddings[filter_indices],
            obs_boxes_3d[filter_indices],
            fps,
        )

        self.track_memory.update(data)

        tracks = self.track_memory.frames[-1]

        # handle vanished tracklets
        cur_memory = self.track_memory.get_current_tracks(
            device=track_ids.device
        )
        for i, track_id in enumerate(cur_memory.track_ids):
            if frame_id > cur_memory.last_frames[i] and track_id > -1:
                pd_box_3d = self._motion_predict(
                    cur_memory, i, track_id, boxes_2d.device
                )

                _, fids = self.track_memory.get_track(track_id)

                new_box_3d = list(cur_memory.boxes_3d)[i]
                new_box_3d[:6] = pd_box_3d[:6]
                new_box_3d[8] = pd_box_3d[6]
                self._update_memory(fids[-1], track_id, "boxes_3d", new_box_3d)

        # update 3D score
        track_scores_3d = tracks.scores_3d * tracks.scores

        return Track3DOut(
            tracks.boxes_3d,
            tracks.class_ids,
            track_scores_3d,
            tracks.track_ids,
        )

    def forward(
        self,
        features: list[torch.Tensor],
        boxes_2d: list[torch.Tensor],
        det_scores: list[torch.Tensor],
        det_boxes_3d: list[torch.Tensor],
        det_scores_3d: list[torch.Tensor],
        det_class_ids: list[torch.Tensor],
        frame_ids: list[int],
        extrinsics: torch.Tensor,
        class_range_map: None | torch.Tensor = None,
        fps: int = 2,
    ) -> Track3DOut:
        """Forward function."""
        assert frame_ids is not None, "Need frame ids during inference!"
        return self._forward_test(
            features,
            boxes_2d,
            det_scores,
            det_boxes_3d,
            det_scores_3d,
            det_class_ids,
            frame_ids,
            extrinsics,
            class_range_map,
            fps,
        )


class FasterRCNNCC3DT(nn.Module):
    """CC-3DT with Faster-RCNN detector."""

    def __init__(
        self,
        num_classes: int,
        backbone: str,
        motion_model: str,
        pure_det: bool,
        class_range_map: None | Tensor = None,
        dataset_fps: int = 2,
        weights: None | str = None,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        anchor_generator = get_default_anchor_generator()
        rpn_bbox_encoder = get_default_rpn_box_encoder()
        rcnn_bbox_encoder = get_default_rcnn_box_encoder()
        roi_head = get_default_roi_head(num_classes)

        self.backbone = ResNet(backbone, pretrained=True, trainable_layers=3)
        self.fpn = FPN(self.backbone.out_channels[2:], 256)
        self.faster_rcnn_heads = FasterRCNNHead(
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            rpn_box_encoder=rpn_bbox_encoder,
            rcnn_box_encoder=rcnn_bbox_encoder,
            roi_head=roi_head,
        )
        self.roi2det = RoI2Det(rcnn_bbox_encoder)
        self.bbox_3d_head = QD3DTBBox3DHead(num_classes=num_classes)
        self.track = CC3DTrack(motion_model=motion_model, pure_det=pure_det)

        self.class_range_map = class_range_map
        self.dataset_fps = dataset_fps

        if weights is not None:
            load_model_checkpoint(self, weights, map_location="cpu")

    def forward(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        frame_ids: list[int],
    ) -> list[CC3DTrackState]:
        """Forward."""
        # TODO implement forward_train
        return self._forward_test(
            images, images_hw, intrinsics, extrinsics, frame_ids
        )

    def _forward_test(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        frame_ids: list[int],
    ) -> list[CC3DTrackState]:
        """Forward inference stage."""
        features = self.backbone(images)
        features = self.fpn(features)
        _, roi, proposals, _, _, _ = self.faster_rcnn_heads(
            features, images_hw
        )

        boxes_2d, scores_2d, class_ids = self.roi2det(
            *roi, proposals.boxes, images_hw
        )

        boxes_3d, scores_3d = self.bbox_3d_head(
            features, boxes_2d, class_ids, intrinsics
        )

        if self.class_range_map is not None:
            self.class_range_map.to(images.device)

        outs = self.track(
            features,
            boxes_2d,
            scores_2d,
            boxes_3d,
            scores_3d,
            class_ids,
            frame_ids,
            extrinsics,
            self.class_range_map,
            self.dataset_fps,
        )
        return outs

    def __call__(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        frame_ids: list[int],
    ) -> list[CC3DTrackState]:
        """Type definition for call implementation."""
        return self._call_impl(
            images, images_hw, intrinsics, extrinsics, frame_ids
        )
