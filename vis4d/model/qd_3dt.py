"""Quasi-dense 3D Tracking model."""
from typing import List

import torch

from ..struct import (
    Boxes2D,
    Boxes3D,
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
)
from .base import BaseModelConfig
from .heads.roi_head import BaseRoIHeadConfig, QD3DTBBox3DHead, build_roi_head
from .qdtrack import QDTrack, QDTrackConfig
from .track.graph import build_track_graph
from .track.utils import split_key_ref_inputs


class QD3DTConfig(QDTrackConfig):
    """Config for quasi-dense 3D tracking model."""

    bbox_3d_head: BaseRoIHeadConfig


class QD3DT(QDTrack):
    """QD-3DT model class."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg = QD3DTConfig(**cfg.dict())
        assert self.cfg.category_mapping is not None
        self.cfg.bbox_3d_head.num_classes = len(self.cfg.category_mapping)  # type: ignore # pylint: disable=line-too-long
        self.bbox_3d_head: QD3DTBBox3DHead = build_roi_head(
            self.cfg.bbox_3d_head
        )
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}

    def forward_train(
        self,
        batch_inputs: List[InputSample],
    ) -> LossesType:
        """Forward function for training."""
        key_inputs, ref_inputs = split_key_ref_inputs(batch_inputs)
        key_targets = key_inputs.targets

        # from vis4d.vis.image import imshow_correspondence
        # for batch_i, key_inp in enumerate(key_inputs):
        #     for ref_inp in ref_inputs:
        #         imshow_correspondence(
        #             key_inp.images.tensor[0],
        #             key_inp.extrinsics[0],
        #             key_inp.intrinsics[0],
        #             ref_inp.images.tensor[batch_i],
        #             ref_inp.extrinsics[batch_i],
        #             ref_inp.intrinsics[batch_i],
        #             key_inp.points.tensor[0],
        #             key_inp.points_extrinsics[0],
        #         )

        # from vis4d.vis.image import show_pointcloud
        # for batch_i, key_inp in enumerate(key_inputs):
        #     show_pointcloud(
        #         key_inp.points.tensor[0],
        #         key_inp.points_extrinsics[0],
        #         key_inp.extrinsics[0],
        #         key_inp.targets.boxes3d[0],
        #     )

        # from vis4d.vis.image import imshow_bboxes3d
        # for batch_i, key_inp in enumerate(key_inputs):
        #     imshow_bboxes3d(
        #         key_inp.images.tensor[0],
        #         key_inp.targets.boxes3d[0],
        #         key_inp.intrinsics[0],
        #     )
        #     for ref_i, ref_inp in enumerate(ref_inputs):
        #         imshow_bboxes3d(
        #             ref_inp[batch_i].images.tensor[0],
        #             ref_inp[batch_i].targets.boxes3d[0],
        #             ref_inp[batch_i].intrinsics[0],
        #         )

        # feature extraction
        key_x = self.detector.extract_features(key_inputs)
        ref_x = [self.detector.extract_features(inp) for inp in ref_inputs]

        losses, key_proposals, _ = self._run_heads_train(
            key_inputs, ref_inputs, key_x, ref_x
        )

        # 3d bbox head
        loss_bbox_3d, _ = self.bbox_3d_head(
            key_inputs, key_proposals, key_x, key_targets
        )
        losses.update(loss_bbox_3d)
        return losses

    def forward_test(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Compute qd-3dt output during inference."""
        assert len(batch_inputs[0]) == 1, "Currently only BS = 1 supported!"

        # if there is more than one InputSample, we switch to multi-sensor:
        # 1st elem is group, rest are sensor frames
        group = batch_inputs[0].to(self.device)
        if len(batch_inputs) > 1:
            frames = InputSample.cat(batch_inputs[1:])
        else:
            frames = batch_inputs[0]

        # detector
        feat = self.detector.extract_features(frames)
        proposals = self.detector.generate_proposals(frames, feat)

        boxes2d_list, _ = self.detector.generate_detections(
            frames, feat, proposals
        )

        # 3d head
        boxes3d_list = self.bbox_3d_head(frames, boxes2d_list, feat)

        # similarity head
        embeddings_list = self.similarity_head(frames, boxes2d_list, feat)

        for inp, boxes2d in zip(frames, boxes2d_list):
            assert inp.metadata[0].size is not None
            input_size = (
                inp.metadata[0].size.width,
                inp.metadata[0].size.height,
            )
            boxes2d.postprocess(
                input_size,
                inp.images.image_sizes[0],
                self.detector.cfg.clip_bboxes_to_image,
            )

        boxes2d = Boxes2D.merge(boxes2d_list)

        for idx, boxes3d in enumerate(boxes3d_list):
            assert isinstance(boxes3d, Boxes3D)
            boxes3d.transform(frames[idx].extrinsics)
        boxes3d = Boxes3D.merge(boxes3d_list)

        embeds = torch.cat(embeddings_list)

        # post-processing
        # boxes_2d = torch.empty(
        #     (0, boxes2d.boxes.shape[1]), device=boxes2d.device
        # )
        # boxes_3d = torch.empty(
        #     (0, boxes3d.boxes.shape[1]), device=boxes3d.device
        # )
        # embeds_post = torch.empty((0, embeds.shape[1]), device=embeds.device)
        # class_ids = torch.empty((0), device=boxes2d.device)

        # boxes3d_post = Boxes3D(boxes_3d, class_ids)
        # boxes2d_post = Boxes2D(boxes_2d, class_ids)

        # for idx, (box2d, box3d) in enumerate(zip(boxes2d, boxes3d)):
        #     boxes3d[idx].boxes[:, -1] *= boxes2d[idx].score
        #     nms_flag = 0
        #     if box2d.class_ids[0] == 5:
        #         nms_dist = 1
        #     else:
        #         nms_dist = 2
        #     for i, (box2d_post, box3d_post) in enumerate(
        #         zip(boxes2d_post, boxes3d_post)
        #     ):
        #         if box2d_post.class_ids == box2d.class_ids:
        #             if (
        #                 torch.cdist(box3d.center, box3d_post.center, p=2)
        #                 <= nms_dist
        #                 and boxes3d[idx].boxes[:, -1] > box3d_post.boxes[:, -1] # pylint: disable=line-too-long
        #             ):
        #                 nms_flag = 1
        #                 boxes_3d[i] = box3d.boxes
        #                 boxes_2d[i] = box2d.boxes
        #                 embeds_post[i] = embeds[idx]
        #                 break
        #     if nms_flag == 0:
        #         boxes_3d = torch.cat([boxes_3d, box3d.boxes])
        #         boxes_2d = torch.cat([boxes_2d, box2d.boxes])
        #         class_ids = torch.cat([class_ids, box2d.class_ids[0]])
        #         embeds_post = torch.cat(
        #             [embeds_post, embeds[idx].unsqueeze(0)]
        #         )

        #     boxes3d_post = Boxes3D(boxes_3d, class_ids)
        #     boxes2d_post = Boxes2D(boxes_2d, class_ids)

        # boxes2d = boxes2d_post
        # boxes3d = boxes3d_post
        # embeds = embeds_post

        boxes_2d = boxes2d.to(torch.device("cpu")).to_scalabel(
            self.cat_mapping
        )

        # associate detections, update graph
        predictions = LabelInstances([boxes2d], [boxes3d])
        tracks = self.track_graph(frames[0], predictions, embeddings=[embeds])

        tracks.boxes3d[0].transform(group.extrinsics.inverse())

        tracks_2d = (
            tracks.boxes2d[0]
            .to(torch.device("cpu"))
            .to_scalabel(self.cat_mapping)
        )
        tracks_3d = (
            tracks.boxes3d[0]
            .to(torch.device("cpu"))
            .to_scalabel(self.cat_mapping)
        )
        return dict(
            detect=[boxes_2d],
            track=[tracks_2d],
            detect_3d=[tracks_3d],
            track_3d=[tracks_3d],
        )
