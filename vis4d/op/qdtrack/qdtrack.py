"""Quasi-dense instance similarity learning model."""
from typing import Dict, List, Optional, Tuple

import torch
from pytorch_lightning.utilities.cli import instantiate_class
from torch import nn

from vis4d.common.bbox.matchers import MaxIoUMatcher
from vis4d.common.bbox.samplers import (
    CombinedSampler,
    match_and_sample_proposals,
)
from vis4d.common.data_pipelines import default as default_augs
from vis4d.op.optimize import DefaultOptimizer
from vis4d.op.track.graph import AssociateQDTrack
from vis4d.op.track.graph.qdtrack import QDTrackMemory, QDTrackState
from vis4d.op.track.similarity.qdtrack import (
    QDSimilarityHead,
    QDTrackInstanceSimilarityLoss,
)
from vis4d.struct import ArgsType

try:
    from mmdet.core.bbox.assigners import SimOTAAssigner

    MMDET_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMDET_INSTALLED = False


class QDTrack(nn.Module):  # TODO remove from op
    """QDTrack model - quasi-dense instance similarity learning."""

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
        self.track_graph = AssociateQDTrack()
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

    def debug_logging(self, logger) -> Dict[str, torch.Tensor]:
        """Logging for debugging"""
        # from vis4d.vis.track import imshow_bboxes
        # for ref_inp, ref_props in zip(ref_inputs, ref_proposals):
        #     for ref_img, ref_prop in zip(ref_inp.images, ref_props):
        #         _, topk_i = torch.topk(ref_prop.boxes[:, -1], 100)
        #         imshow_bboxes(ref_img.tensor[0], ref_prop[topk_i])
        # for batch_i, key_inp in enumerate(key_inputs):
        #    imshow_bboxes(
        #        key_inp.images.tensor[0], key_inp.targets.boxes2d[0]
        #    )
        #    for ref_i, ref_inp in enumerate(ref_inputs):
        #        imshow_bboxes(
        #            ref_inp[batch_i].images.tensor[0],
        #            ref_inp[batch_i].targets.boxes2d[0],
        #        )

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
            # reset graph at begin of sequence TODO move outside
            if frame_id == 0:
                self.track_memory.reset()

            cur_memory = self.track_memory.get_current_tracks(box.device)
            track_ids, filter_indcs = self.track_graph(
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
                box[filter_indcs],
                score[filter_indcs],
                cls_id[filter_indcs],
                embeds[filter_indcs],
            )
            self.track_memory.update(data)
            batched_tracks.append(self.track_memory.last_frame)

        return batched_tracks


class ClippedSimOTAAssigner(SimOTAAssigner):  # type: ignore
    """Modified SimOTAAssigner to support boxes with center outside of img."""

    def __init__(self, h: int, w: int, *args, **kwargs) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        self.im_h, self.im_w = h, w

    def get_in_gt_and_in_center_info(self, priors, gt_bboxes):
        """Compute labels for classification branch."""
        num_gt = gt_bboxes.size(0)

        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        # is prior centers in gt bboxes, shape: [n_prior, n_gt]
        l_ = repeated_x - gt_bboxes[:, 0]
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y

        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        gt_cxs = torch.clamp(
            (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0, min=0, max=self.im_w
        )
        gt_cys = torch.clamp(
            (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0, min=0, max=self.im_h
        )

        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = (
            is_in_gts[is_in_gts_or_centers, :]
            & is_in_cts[is_in_gts_or_centers, :]
        )
        return is_in_gts_or_centers, is_in_boxes_and_centers


class QDTrackYOLOX(QDTrack):
    """QDTrack + YOLOX detector."""

    def __init__(
        self,
        *args: ArgsType,
        im_hw: Tuple[int, int] = (800, 1440),
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        assert MMDET_INSTALLED, "QDTrackYOLOX needs mmdet installed!"
        self.im_hw = im_hw
        if self.detector.bbox_head.mm_dense_head.train_cfg:
            assign_args = (
                self.detector.bbox_head.mm_dense_head.train_cfg.assigner
            )
            assign_args.pop("type")
            self.detector.bbox_head.mm_dense_head.assigner = (
                ClippedSimOTAAssigner(*im_hw, **assign_args)
            )


class YOLOXOptimize(DefaultOptimizer):
    """QDTrack + YOLOX detector optimization routine."""

    def __init__(
        self,
        *args: ArgsType,
        no_aug_epochs: int = 10,
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        self.no_aug_epochs = no_aug_epochs
        assert hasattr(
            self.model, "im_hw"
        ), "Need image hw to reset augmentations"
        self.im_hw = self.model.im_hw

    def on_train_epoch_start(self):
        """In the last training epochs: add L1 loss, turn off augmentations."""
        if self.current_epoch >= self.trainer.max_epochs - self.no_aug_epochs:
            self.detector.bbox_head.mm_dense_head.use_l1 = True
            self.trainer.datamodule.train_datasets.transformations = (
                default_augs(self.im_hw)
            )
            self.trainer.reset_train_dataloader(self)

    def configure_optimizers(self):
        """Configure optimizers and schedulers of model."""
        params = []
        for name, param in self.named_parameters():
            param_group = {"params": [param]}
            if "bias" in name or "norm" in name:
                param_group["weight_decay"] = 0.0
            params.append(param_group)
        optimizer = instantiate_class(params, self.optimizer_init)
        scheduler = instantiate_class(optimizer, self.lr_scheduler_init)
        return [optimizer], [scheduler]
