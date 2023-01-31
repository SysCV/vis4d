"""CC-3DT model implementation.

This file composes the operations associated with
CC-3DT `https://arxiv.org/abs/2212.01247' into the full model implementation.
"""
import torch

from typing import List

from torch import nn

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
from vis4d.state.track.qd_3dt import QD3DTrackState, QD3DTrackMemory
from vis4d.op.track.cc_3dt import CC3DTrackAssociation
from vis4d.op.detect_3d.qd_3dt import QD3DTBBox3DHead
from vis4d.common import ArgsType
from vis4d.op.track.motion.kf3d import KF3DMotionModel
from vis4d.op.detect_3d.filter import filter_distance, bev_3d_nms

from vis4d.op.geometry.transform import transform_points
from vis4d.op.geometry.rotation import rotate_orientation, rotate_velocities

from vis4d.op.detect.faster_rcnn import AnchorGenerator
import pdb
from vis4d.data.transforms.normalize import normalize


def get_default_anchor_generator() -> AnchorGenerator:
    """Get default anchor generator."""
    return AnchorGenerator(
        scales=[4, 8],
        ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
        strides=[4, 8, 16, 32, 64],
    )


class QD3DTrack(QDTrack):
    """QD-3DT model."""

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
        self.track_memory = QD3DTrackMemory(memory_limit=memory_size)
        self.track_graph = CC3DTrackAssociation()
        self.motion_model = motion_model
        self.motion_dims = motion_dims
        self.num_frames = num_frames
        self.pure_det = pure_det
        self.fps = fps

    def forward(
        self,
        features: list[torch.Tensor],
        det_boxes: list[torch.Tensor],
        det_scores: list[torch.Tensor],
        det_boxes_3d: list[torch.Tensor],
        det_scores_3d: list[torch.Tensor],
        det_class_ids: list[torch.Tensor],
        frame_ids: List[int],
        extrinsics: torch.Tensor,
    ) -> list[QD3DTrackState]:
        """Forward function."""
        assert frame_ids is not None, "Need frame ids during inference!"
        return self._forward_test(
            features,
            det_boxes,
            det_scores,
            det_boxes_3d,
            det_scores_3d,
            det_class_ids,
            frame_ids,
            extrinsics,
        )

    def _forward_test(
        self,
        features_list: list[torch.Tensor],
        det_boxes_list: list[torch.Tensor],
        det_scores_list: list[torch.Tensor],
        det_boxes_3d_list: list[torch.Tensor],
        det_scores_3d_list: list[torch.Tensor],
        det_class_ids_list: list[torch.Tensor],
        frame_ids: List[int],
        extrinsics: torch.Tensor,
    ) -> list[QD3DTrackState]:
        """Forward function during testing."""
        embeddings_list = list(
            self.similarity_head(features_list, det_boxes_list)
        )

        # pdb.set_trace()

        # TODO: add class range automatically
        class_range_map = torch.Tensor(
            [40, 40, 40, 50, 50, 50, 50, 50, 50, 30, 30]
        ).to(det_boxes_list[0].device)

        if sum(len(b) for b in det_boxes_3d_list) != 0:
            for i, det_boxes_3d in enumerate(det_boxes_3d_list):
                valid_boxes = filter_distance(
                    det_class_ids_list[i], det_boxes_3d, class_range_map
                )

                # move 3D boxes to world coordinates
                det_boxes_3d_list[i] = det_boxes_3d[valid_boxes]
                det_boxes_3d_list[i][:, :3] = transform_points(
                    det_boxes_3d_list[i][:, :3], extrinsics[i]
                )
                det_boxes_3d_list[i][:, 6:9] = rotate_orientation(
                    det_boxes_3d_list[i][:, 6:9], extrinsics[i]
                )
                det_boxes_3d_list[i][:, 9:12] = rotate_velocities(
                    det_boxes_3d_list[i][:, 9:12], extrinsics[i]
                )

                det_boxes_list[i] = det_boxes_list[i][valid_boxes]
                det_scores_list[i] = det_scores_list[i][valid_boxes]
                det_scores_3d_list[i] = det_scores_3d_list[i][valid_boxes]
                det_class_ids_list[i] = det_class_ids_list[i][valid_boxes]
                embeddings_list[i] = embeddings_list[i][valid_boxes]

        det_boxes = torch.cat(det_boxes_list)
        det_scores = torch.cat(det_scores_list)
        det_boxes_3d = torch.cat(det_boxes_3d_list)
        det_scores_3d = torch.cat(det_scores_3d_list)
        det_class_ids = torch.cat(det_class_ids_list)
        embeddings = torch.cat(embeddings_list)

        if self.pure_det:
            return

        # add camera id
        camera_ids_list = [
            torch.ones(
                det_boxes_list[i].shape[0],
                device=det_boxes_list[i].device,
            )
            * i
            for i in range(len(det_boxes_list))
        ]
        camera_ids = torch.cat(camera_ids_list)

        # merge multi-view boxes
        keep_indices = bev_3d_nms(
            det_boxes_3d,
            det_scores * det_scores_3d,
            det_class_ids,
        )

        det_boxes = det_boxes[keep_indices]
        det_scores = det_scores[keep_indices]
        det_boxes_3d = det_boxes_3d[keep_indices]
        det_scores_3d = det_scores_3d[keep_indices]
        det_class_ids = det_class_ids[keep_indices]
        embeddings = embeddings[keep_indices]
        camera_ids = camera_ids[keep_indices]

        # TODO: add bateched tracks with cc-3dt data connector, currently only
        # support single batch per GPU.
        frame_id = frame_ids[0]
        tracks = []

        # reset graph at begin of sequence
        if frame_id == 0:
            self.track_memory.reset()
            TrackIDCounter.reset()

        cur_memory = self.track_memory.get_current_tracks(det_boxes.device)

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
                (0, 7), device=det_boxes.device
            )

        obs_boxes_3d = torch.cat(
            [det_boxes_3d[:, :6], det_boxes_3d[:, 8].unsqueeze(1)], dim=1
        )

        track_ids, match_ids, filter_indices = self.track_graph(
            det_boxes,
            det_scores,
            obs_boxes_3d,
            det_scores_3d,
            det_class_ids,
            embeddings,
            camera_ids,
            memory_boxes_3d,
            cur_memory.track_ids,
            cur_memory.class_ids,
            cur_memory.embeddings,
            memory_boxes_3d_predict,
            cur_memory.velocities,
        )

        # motion model & velocity
        motion_models = []
        velocities = []
        for i, track_id in enumerate(track_ids):
            bbox_3d = det_boxes_3d[filter_indices][i]
            info = det_scores_3d[filter_indices][i].unsqueeze(0)
            obs_3d = torch.cat([obs_boxes_3d[filter_indices][i], info], dim=0)
            if track_id in match_ids:
                # update track
                indice = (cur_memory.track_ids == track_id).nonzero(
                    as_tuple=False
                )[0]
                motion_model = cur_memory.motion_models[indice]

                time_since_update = motion_model.time_since_update

                motion_model.update(obs_3d)
                pd_box_3d = motion_model.get_state()[: self.motion_dims]

                det_boxes_3d[filter_indices][i][:6] = pd_box_3d[:6]
                det_boxes_3d[filter_indices][i][8] = pd_box_3d[6]

                det_boxes_3d[filter_indices][i][9:12] = (
                    motion_model.predict_velocity() * self.fps
                )

                prev_obs = memory_boxes_3d[indice]
                velocities.append(
                    (
                        (pd_box_3d - prev_obs[: self.motion_dims])
                        / time_since_update
                    ).squeeze(0)
                )
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

        velocities = torch.stack(velocities, dim=0)

        data = QD3DTrackState(
            track_ids,
            det_boxes[filter_indices],
            det_scores[filter_indices],
            det_boxes_3d[filter_indices],
            det_scores_3d[filter_indices],
            det_class_ids[filter_indices],
            embeddings[filter_indices],
            motion_models,
            velocities,
        )
        self.track_memory.update(data)

        tracks = self.track_memory.last_frame

        # Update 3D score and move 3D boxes into group sensor coordinate
        track_scores_3d = tracks.scores_3d * tracks.scores

        return QD3DTrackState(
            tracks.track_ids,
            tracks.boxes,
            tracks.scores,
            tracks.boxes_3d,
            track_scores_3d,
            tracks.class_ids,
            tracks.embeddings,
            tracks.motion_models,
            tracks.velocities,
        )


class FasterRCNNCC3DT(nn.Module):
    """CC-3DT with Faster-RCNN detector."""

    def __init__(self, num_classes: int) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.anchor_gen = get_default_anchor_generator()
        self.rpn_bbox_encoder = get_default_rpn_box_encoder()
        self.rcnn_bbox_encoder = get_default_rcnn_box_encoder()

        self.backbone = ResNet("resnet50", pretrained=True, trainable_layers=3)
        self.fpn = FPN(self.backbone.out_channels[2:], 256)
        self.faster_rcnn_heads = FasterRCNNHead(
            num_classes=num_classes,
            anchor_generator=self.anchor_gen,
            rpn_box_encoder=self.rpn_bbox_encoder,
            rcnn_box_encoder=self.rcnn_bbox_encoder,
        )
        self.roi2det = RoI2Det(self.rcnn_bbox_encoder)
        self.bbox_3d_head = QD3DTBBox3DHead(num_classes=num_classes)
        self.track = QD3DTrack()

    def forward(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        frame_ids: list[int],
    ) -> list[QD3DTrackState]:
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
    ) -> list[QD3DTrackState]:
        """Forward inference stage."""
        images = normalize(images)
        features = self.backbone(images)
        features = self.fpn(features)
        detector_out = self.faster_rcnn_heads(features, images_hw)

        boxes, scores, class_ids = self.roi2det(
            *detector_out.roi, detector_out.proposals.boxes, images_hw
        )

        pdb.set_trace()

        boxes_3d, depth_uncertainty = self.bbox_3d_head(
            features, boxes, class_ids, intrinsics
        )

        outs = self.track(
            features,
            boxes,
            scores,
            boxes_3d,
            depth_uncertainty,
            class_ids,
            frame_ids,
            extrinsics,
        )
        return outs

    def __call__(
        self,
        images: torch.Tensor,
        images_hw: list[tuple[int, int]],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        frame_ids: list[int],
    ) -> list[QD3DTrackState]:
        """Type definition for call implementation."""
        return self._call_impl(
            images, images_hw, intrinsics, extrinsics, frame_ids
        )
