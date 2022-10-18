"""Quasi-dense instance similarity learning model."""
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from vis4d.model.detect.faster_rcnn import FasterRCNN, FasterRCNNLoss
from vis4d.op.detect.rcnn import DetOut, RCNNLoss, RCNNLosses, RoI2Det
from vis4d.op.detect.rpn import RPNLoss, RPNLosses
from vis4d.op.fpp.fpn import FPN
from vis4d.op.track.graph.assignment import TrackIDCounter
from vis4d.op.track.qdtrack import (
    QDSimilarityHead,
    QDTrackAssociation,
    QDTrackInstanceSimilarityLoss,
)
from vis4d.state.track.qdtrack import QDTrackMemory, QDTrackState

REV_KEYS = [
    (r"^detector.rpn_head.mm_dense_head\.", "rpn_head."),
    ("\.rpn_reg\.", ".rpn_box."),
    (r"^detector.roi_head.mm_roi_head.bbox_head\.", "roi_head."),
    (r"^detector.backbone.mm_backbone\.", "body."),
    (
        r"^detector.backbone.neck.mm_neck.lateral_convs\.",
        "inner_blocks.",
    ),
    (
        r"^detector.backbone.neck.mm_neck.fpn_convs\.",
        "layer_blocks.",
    ),
    ("\.conv.weight", ".weight"),
    ("\.conv.bias", ".bias"),
]


def _debug_visualization(key_inputs, ref_inputs, ref_proposals):  # TODO revise
    from vis4d.vis.track import imshow_bboxes

    for ref_inp, ref_props in zip(ref_inputs, ref_proposals):
        for ref_img, ref_prop in zip(ref_inp.images, ref_props):
            _, topk_i = torch.topk(ref_prop.boxes[:, -1], 100)
            imshow_bboxes(ref_img.tensor[0], ref_prop[topk_i])
    for batch_i, key_inp in enumerate(key_inputs):
        imshow_bboxes(key_inp.images.tensor[0], key_inp.targets.boxes2d[0])
        for ref_i, ref_inp in enumerate(ref_inputs):
            imshow_bboxes(
                ref_inp[batch_i].images.tensor[0],
                ref_inp[batch_i].targets.boxes2d[0],
            )


class QDTrack(nn.Module):
    """QDTrack - quasi-dense instance similarity learning."""

    def __init__(
        self,
        memory_size: int = 10,
        num_ref_views: int = 1,
        proposal_append_gt: bool = True,
    ) -> None:
        """Init."""
        super().__init__()
        self.num_ref_views = num_ref_views
        self.similarity_head = QDSimilarityHead()

        # only in inference
        self.track_graph = QDTrackAssociation()
        self.track_memory = QDTrackMemory(memory_limit=memory_size)

        self.box_sampler = CombinedSampler(
            batch_size=256,
            positive_fraction=0.5,
            pos_strategy="instance_balanced",
            neg_strategy="iou_balanced",
        )

        self.box_matcher = MaxIoUMatcher(
            thresholds=[0.3, 0.7],
            labels=[0, -1, 1],
            allow_low_quality_matches=False,
        )
        self.proposal_append_gt = proposal_append_gt
        self.track_loss = QDTrackInstanceSimilarityLoss()

    def forward(
        self,
        features: List[torch.Tensor],
        det_boxes: List[torch.Tensor],
        det_scores: List[torch.Tensor],
        det_class_ids: List[torch.Tensor],
        frame_ids: Optional[Tuple[int, ...]] = None,
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_track_ids: Optional[List[torch.Tensor]] = None,
    ) -> List[QDTrackState]:
        """Forward function."""
        if target_boxes is not None:
            assert (
                target_track_ids is not None
            ), "Need targets during training!"
            return self._forward_train(
                features,
                det_boxes,
                target_boxes,
                target_track_ids,
            )
        assert frame_ids is not None, "Need frame ids during inference!"
        return self._forward_test(
            features, det_boxes, det_scores, det_class_ids, frame_ids
        )

    def _split_views(
        self,
        embeddings: List[torch.Tensor],
        target_track_ids: List[torch.Tensor],
    ) -> Tuple[
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[torch.Tensor],
        List[List[torch.Tensor]],
    ]:
        """Split batch and reference view dimension."""
        B, R = len(embeddings), self.num_ref_views + 1
        key_embeddings = [embeddings[i] for i in range(0, B, R)]
        key_track_ids = [target_track_ids[i] for i in range(0, B, R)]
        ref_embeddings, ref_track_ids = [], []
        for i in range(1, B, R):
            current_refs, current_track_ids = [], []
            for j in range(i, i + R - 1):
                current_refs.append(embeddings[j])
                current_track_ids.append(target_track_ids[j])
            ref_embeddings.append(current_refs)
            ref_track_ids.append(current_track_ids)
        return key_embeddings, ref_embeddings, key_track_ids, ref_track_ids

    @torch.no_grad()
    def _sample_proposals(
        self,
        det_boxes: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
        target_track_ids: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Sample proposals for instance similarity learning."""
        B, R = len(det_boxes), self.num_ref_views + 1

        if self.proposal_append_gt:
            det_boxes = [
                torch.cat([d, t]) for d, t in zip(det_boxes, target_boxes)
            ]

        (
            sampled_box_indices,
            sampled_target_indices,
            sampled_labels,
        ) = match_and_sample_proposals(
            self.box_matcher,
            self.box_sampler,
            det_boxes,
            target_boxes,
        )
        sampled_boxes, sampled_track_ids = [], []
        for i in range(B):
            positives = sampled_labels[i] == 1
            if i % R == 0:  # take only positives for keyframes
                sampled_box = det_boxes[i][sampled_box_indices[i]][positives]
                sampled_tr_id = target_track_ids[i][sampled_target_indices[i]][
                    positives
                ]
            else:  # set track_ids to -1 for all negatives
                sampled_box = det_boxes[i][sampled_box_indices[i]]
                sampled_tr_id = target_track_ids[i][sampled_target_indices[i]]
                sampled_tr_id[~positives] = -1

            sampled_boxes.append(sampled_box)
            sampled_track_ids.append(sampled_tr_id)
        return sampled_boxes, sampled_track_ids

    def _forward_train(
        self,
        features: List[torch.Tensor],
        det_boxes: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
        target_track_ids: List[torch.Tensor],
    ):
        """TODO define return type."""
        sampled_boxes, sampled_track_ids = self._sample_proposals(
            det_boxes, target_boxes, target_track_ids
        )
        embeddings = self.similarity_head(features, sampled_boxes)
        return self.track_loss(
            *self._split_views(embeddings, sampled_track_ids)
        )

    def _forward_test(
        self,
        features: List[torch.Tensor],
        det_boxes: List[torch.Tensor],
        det_scores: List[torch.Tensor],
        det_class_ids: List[torch.Tensor],
        frame_ids: Tuple[int, ...],
    ) -> List[QDTrackState]:
        """Forward during test."""
        embeddings = self.similarity_head(features, det_boxes)

        batched_tracks = []
        for frame_id, box, score, cls_id, embeds in zip(
            frame_ids, det_boxes, det_scores, det_class_ids, embeddings
        ):
            # reset graph at begin of sequence
            if frame_id == 0:
                self.track_memory.reset()
                TrackIDCounter.reset()

            cur_memory = self.track_memory.get_current_tracks(box.device)
            track_ids, filter_indices = self.track_graph(
                box,
                score,
                cls_id,
                embeds,
                cur_memory.track_ids,
                cur_memory.class_ids,
                cur_memory.embeddings,
            )

            data = QDTrackState(
                track_ids,
                box[filter_indices],
                score[filter_indices],
                cls_id[filter_indices],
                embeds[filter_indices],
            )
            self.track_memory.update(data)
            batched_tracks.append(self.track_memory.last_frame)

        return batched_tracks


class FasterRCNNQDTrack(nn.Module):
    """Wrap qdtrack with detector."""

    def __init__(self, num_classes: int) -> None:
        """Init."""
        super().__init__()
        self.faster_rcnn = FasterRCNN(num_classes=num_classes)
        self.qdtrack = QDTrack()

    def forward(
        self,
        images: torch.Tensor,
        images_hw: List[Tuple[int, int]],
        frame_ids: List[int],
    ) -> List[QDTrackState]:
        """Forward."""
        return self._forward_test(images, images_hw, frame_ids)

    def _forward_test(
        self,
        images: torch.Tensor,
        images_hw: List[Tuple[int, int]],
        frame_ids: List[int],
    ) -> List[QDTrackState]:
        """Forward inference stage."""
        features = self.backbone(images)
        features = self.fpn(features)
        detector_out = self.faster_rcnn_heads(features, images_hw)

        boxes, scores, class_ids = self.faster_rcnn.roi2det(
            *detector_out.roi, detector_out.proposals.boxes, images_hw
        )
        outs = self.qdtrack(features, boxes, scores, class_ids, frame_ids)
        return outs
