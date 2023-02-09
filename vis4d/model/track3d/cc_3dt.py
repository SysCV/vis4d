"""CC-3DT model implementation.

This file composes the operations associated with
CC-3DT `https://arxiv.org/abs/2212.01247' into the full model implementation.
"""
import torch
import copy

from typing import List
from vis4d.engine.ckpt import load_model_checkpoint
from torch import nn, Tensor
from vis4d.op.box.box2d import bbox_clip
from vis4d.model.track.qdtrack import QDTrack
from vis4d.op.base import ResNet
from vis4d.op.detect.faster_rcnn import (
    FasterRCNNHead,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.op.detect.rcnn import RoI2Det
from vis4d.op.fpp import FPN
from vis4d.op.track.assignment import TrackIDCounter
from vis4d.state.track.cc_3dt import CC3DTrackState, CC3DTrackMemory
from vis4d.op.track.cc_3dt import CC3DTrackAssociation
from vis4d.op.detect_3d.qd_3dt import QD3DTBBox3DHead
from vis4d.common import ArgsType
from vis4d.op.track.motion.kf3d import KF3DMotionModel
from vis4d.op.detect_3d.filter import filter_distance, bev_3d_nms

from vis4d.op.geometry.transform import transform_points
from vis4d.op.geometry.rotation import rotate_orientation, rotate_velocities

from vis4d.op.detect.faster_rcnn import AnchorGenerator
import pdb


def get_default_anchor_generator() -> AnchorGenerator:
    """Get default anchor generator."""
    return AnchorGenerator(
        scales=[4, 8],
        ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
        strides=[4, 8, 16, 32, 64],
    )


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
        fps: int = 2,
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__(*args, memory_size, **kwargs)
        self.track_memory = CC3DTrackMemory(memory_limit=memory_size)
        self.track_graph = CC3DTrackAssociation()
        self.motion_model = motion_model
        self.motion_dims = motion_dims
        self.num_frames = num_frames
        self.pure_det = pure_det
        self.fps = fps
        self.memo_momentum = 0.8

    def forward(
        self,
        features: list[torch.Tensor],
        boxes_2d: list[torch.Tensor],
        det_scores: list[torch.Tensor],
        det_boxes_3d: list[torch.Tensor],
        det_scores_3d: list[torch.Tensor],
        det_class_ids: list[torch.Tensor],
        frame_ids: List[int],
        extrinsics: torch.Tensor,
        images_hw: list[tuple[int, int]],
    ) -> list[CC3DTrackState]:
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
            images_hw,
        )

    def _cam_to_global(
        self,
        boxes_2d_list: list[Tensor],
        scores_2d_list: list[Tensor],
        boxes_3d_list: list[Tensor],
        scores_3d_list: list[Tensor],
        class_ids_list: list[Tensor],
        embeddings_list: list[Tensor],
        extrinsics: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Move 3D boxes to global coordinates."""
        # TODO: add class range automatically
        class_range_map = torch.Tensor(
            [40, 40, 40, 50, 50, 50, 50, 50, 30, 30]
        ).to(boxes_2d_list[0].device)

        camera_ids_list = []
        if sum(len(b) for b in boxes_3d_list) != 0:
            for i, boxes_3d in enumerate(boxes_3d_list):
                if len(boxes_3d) != 0:
                    # filter out boxes that are too far away
                    valid_boxes = filter_distance(
                        class_ids_list[i], boxes_3d, class_range_map
                    )

                    # move 3D boxes to world coordinates
                    boxes_3d_list[i] = boxes_3d[valid_boxes]
                    boxes_3d_list[i][:, :3] = transform_points(
                        boxes_3d_list[i][:, :3], extrinsics[i]
                    )
                    boxes_3d_list[i][:, 6:9] = rotate_orientation(
                        boxes_3d_list[i][:, 6:9], extrinsics[i]
                    )
                    boxes_3d_list[i][:, 9:12] = rotate_velocities(
                        boxes_3d_list[i][:, 9:12], extrinsics[i]
                    )

                    boxes_2d_list[i] = boxes_2d_list[i][valid_boxes]
                    # add camera id
                    camera_ids_list.append(
                        (torch.ones(len(boxes_2d_list[i])) * i).to(
                            boxes_2d_list[i].device
                        )
                    )

                    scores_2d_list[i] = scores_2d_list[i][valid_boxes]
                    scores_3d_list[i] = scores_3d_list[i][valid_boxes]
                    class_ids_list[i] = class_ids_list[i][valid_boxes]
                    embeddings_list[i] = embeddings_list[i][valid_boxes]

        boxes_2d = torch.cat(boxes_2d_list)
        camera_ids = torch.cat(camera_ids_list)
        scores_2d = torch.cat(scores_2d_list)
        boxes_3d = torch.cat(boxes_3d_list)
        scores_3d = torch.cat(scores_3d_list)
        class_ids = torch.cat(class_ids_list)
        embeddings = torch.cat(embeddings_list)
        return (
            boxes_2d,
            camera_ids,
            scores_2d,
            boxes_3d,
            scores_3d,
            class_ids,
            embeddings,
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
    ):
        """Update track."""
        motion_models = []
        velocities = []
        last_frames = []
        acc_frames = []
        for i, track_id in enumerate(track_ids):
            bbox_3d = boxes_3d[i]
            obs_3d = obs_boxes_3d[i]
            if track_id in match_ids:
                # update track
                track = self.track_memory.get_track(track_id)[-1]
                motion_model = copy.deepcopy(track.motion_models)

                # pdb.set_trace()

                motion_model.update(obs_3d)
                pd_box_3d = motion_model.get_state()[: self.motion_dims]

                boxes_3d[i][:6] = pd_box_3d[:6]
                boxes_3d[i][8] = pd_box_3d[6]

                boxes_3d[i][9:12] = motion_model.predict_velocity() * self.fps

                prev_obs = torch.cat(
                    [track.boxes_3d[0, :6], track.boxes_3d[:, 8]], dim=0
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

                motion_models.append(motion_model)
            else:
                # create track
                motion_models.append(
                    KF3DMotionModel(
                        num_frames=self.num_frames,
                        obs_3d=obs_3d,
                        motion_dims=self.motion_dims,
                    )
                )
                velocities.append(
                    torch.zeros(self.motion_dims, device=bbox_3d.device)
                )
                acc_frames.append(0)
            last_frames.append(frame_id)

        velocities = torch.stack(velocities)
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
            motion_models,
            velocities,
            last_frames,
            acc_frames,
        )

    def _forward_test(
        self,
        features_list: list[torch.Tensor],
        boxes_2d_list: list[torch.Tensor],
        scores_2d_list: list[torch.Tensor],
        boxes_3d_list: list[torch.Tensor],
        scores_3d_list: list[torch.Tensor],
        class_ids_list: list[torch.Tensor],
        frame_ids: List[int],
        extrinsics: torch.Tensor,
        images_hw: list[tuple[int, int]],
    ) -> list[CC3DTrackState]:
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
        ) = self._cam_to_global(
            boxes_2d_list,
            scores_2d_list,
            boxes_3d_list,
            scores_3d_list,
            class_ids_list,
            embeddings_list,
            extrinsics,
        )

        # TODO: Add pure detection mode
        if self.pure_det:
            raise NotImplementedError

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

        # TODO: Add bateched tracks with cc-3dt data connector
        frame_id = frame_ids[0][0]
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
            for ind, motion_model in enumerate(cur_memory.motion_models):
                memory_boxes_3d_predict[ind, :3] += motion_model.predict(
                    update_state=motion_model.age != 0
                )[self.motion_dims :]
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
        )

        self.track_memory.update(data)

        tracks = self.track_memory.last_frame

        # Update 3D score and move 3D boxes into group sensor coordinate
        track_scores_3d = tracks.scores_3d * tracks.scores

        return CC3DTrackState(
            tracks.track_ids,
            tracks.boxes,
            tracks.camera_ids,
            tracks.scores,
            tracks.boxes_3d,
            track_scores_3d,
            tracks.class_ids,
            tracks.embeddings,
            tracks.motion_models,
            tracks.velocities,
            tracks.last_frames,
            tracks.acc_frames,
        )


class FasterRCNNCC3DT(nn.Module):
    """CC-3DT with Faster-RCNN detector."""

    def __init__(
        self,
        num_classes: int,
        backbone: str,
        motion_model: str,
        pure_det: bool,
        weights: None | str = None,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.anchor_gen = get_default_anchor_generator()
        self.rpn_bbox_encoder = get_default_rpn_box_encoder()
        self.rcnn_bbox_encoder = get_default_rcnn_box_encoder()

        self.backbone = ResNet(backbone, pretrained=True, trainable_layers=3)
        self.fpn = FPN(self.backbone.out_channels[2:], 256)
        self.faster_rcnn_heads = FasterRCNNHead(
            num_classes=num_classes,
            anchor_generator=self.anchor_gen,
            rpn_box_encoder=self.rpn_bbox_encoder,
            rcnn_box_encoder=self.rcnn_bbox_encoder,
        )
        self.roi2det = RoI2Det(self.rcnn_bbox_encoder)
        self.bbox_3d_head = QD3DTBBox3DHead(num_classes=num_classes)
        self.track = CC3DTrack(motion_model=motion_model, pure_det=pure_det)

        if weights is not None:
            load_model_checkpoint(self, weights)

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

        outs = self.track(
            features,
            boxes_2d,
            scores_2d,
            boxes_3d,
            scores_3d,
            class_ids,
            frame_ids,
            extrinsics,
            images_hw,
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
