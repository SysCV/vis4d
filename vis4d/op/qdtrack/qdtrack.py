"""Quasi-dense instance similarity learning model."""
from typing import Dict, List, Optional, Tuple

import torch
from pytorch_lightning.utilities.cli import instantiate_class
from torch import nn

from vis4d.common.bbox.matchers import MaxIoUMatcher
from vis4d.common.bbox.samplers import CombinedSampler
from vis4d.common.data_pipelines import default as default_augs
from vis4d.op.detect import FasterRCNN
from vis4d.op.heads.roi_head.rcnn import RoI2Det
from vis4d.op.optimize import DefaultOptimizer
from vis4d.op.track.graph import QDTrackGraph
from vis4d.op.track.graph.qdtrack import Tracks
from vis4d.op.track.similarity import QDSimilarityHead
from vis4d.struct import ArgsType

try:
    from mmdet.core.bbox.assigners import SimOTAAssigner

    MMDET_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMDET_INSTALLED = False


class QDTrack(nn.Module):
    """QDTrack model - quasi-dense instance similarity learning."""

    def __init__(
        self, detector: FasterRCNN, detector_transform: RoI2Det
    ) -> None:
        """Init."""
        super().__init__()
        self.detector = detector
        self.detector_transform = detector_transform
        self.similarity_head = QDSimilarityHead()
        self.track_graph = QDTrackGraph()
        self.track_memory = Tracks(memory_limit=10)
        self.backdrop_memory = Tracks(memory_limit=10)

        self.sampler = CombinedSampler(
            batch_size=256,
            positive_fraction=0.5,
            pos_strategy="instance_balanced",
            neg_strategy="iou_balanced",
        )

        self.matcher = MaxIoUMatcher(
            thresholds=[0.3, 0.7],
            labels=[0, -1, 1],
            allow_low_quality_matches=False,
        )

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
        images: torch.Tensor,
        frame_ids: Optional[Tuple[int, ...]] = None,
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
        target_track_ids: Optional[List[torch.Tensor]] = None,
    ) -> List[Tuple[torch.Tensor, ...]]:  # TODO define return type
        """Forward function."""
        if self.training:  # TODO change to targets existing
            assert (
                target_boxes is not None
                and target_classes is not None
                and target_track_ids is not None
            ), "Need targets during training!"
            return self._forward_train(
                images, target_boxes, target_classes, target_track_ids
            )
        assert frame_ids is not None, "Need frame ids during inference!"
        return self._forward_test(images, frame_ids)

    def _forward_train(
        images: torch.Tensor,
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
        target_track_ids: List[torch.Tensor],
    ):
        """TODO define return type."""

        # # TODO will be part of training loop
        # sampling_results, sampled_boxes, sampled_targets = [], [], []
        # for i, (box, tgt) in enumerate(zip(boxes, targets)):
        #     sampling_result = match_and_sample_proposals(
        #         self.matcher,
        #         self.sampler,
        #         box,
        #         tgt.boxes2d,
        #         self.proposal_append_gt,
        #     )
        #     sampling_results.append(sampling_result)
        #
        #     sampled_box = sampling_result.sampled_boxes
        #     sampled_tgt = sampling_result.sampled_targets
        #     positives = [l == 1 for l in sampling_result.sampled_labels]
        #     if i == 0:  # take only positives for keyframe (assumed at i=0)
        #         sampled_box = [b[p] for b, p in zip(sampled_box, positives)]
        #         sampled_tgt = [t[p] for t, p in zip(sampled_tgt, positives)]
        #     else:  # set track_ids to -1 for all negatives
        #         for pos, samp_tgt in zip(positives, sampled_tgt):
        #             samp_tgt.track_ids[~pos] = -1
        #
        #     sampled_boxes.append(sampled_box)
        #     sampled_targets.append(sampled_tgt)

    def _forward_test(
        self, images: torch.Tensor, frame_ids: Tuple[int, ...]
    ):  # TODO input detections, backbone outs, tracks [inference], no images
        """Forward during test."""
        # detection
        # detector_out = self.detector(images)
        # detections = self.detector_transform(
        #     detector_out.roi_cls_out,
        #     detector_out.roi_reg_out,
        #     detector_out.proposal_boxes,
        #     images.shape,
        # )
        # boxes, scores, class_ids = (
        #     [d.boxes for d in detections],
        #     [d.scores for d in detections],
        #     [d.class_ids for d in detections],
        # )

        # similarity head
        embeddings = self.similarity_head(detector_out.backbone_out[:4], boxes)

        batched_tracks = []
        for frame_id, box, score, cls_id, embeds in zip(
            frame_ids, boxes, scores, class_ids, embeddings
        ):
            # # reset graph at begin of sequence
            # if frame_id == 0:
            #     self.track_memory = Tracks(memory_limit=10)
            #     self.backdrop_memory = Tracks(memory_limit=10)

            # tracks = self.track_memory.get_frames(
            #     max(0, frame_id - 10), frame_id
            # )
            # backdrops = self.backdrop_memory.get_frames(
            #     max(0, frame_id - 1), frame_id
            # )

            track_ids, filter_indcs = self.track_graph(
                box, score, cls_id, embeds, tracks, backdrops
            )

            # data = (
            #     box[filter_indcs],
            #     score[filter_indcs],
            #     cls_id[filter_indcs],
            #     embeds[filter_indcs],
            # )
            # valid_tracks = track_ids != -1
            # new_tracks = (
            #     track_ids[valid_tracks],
            #     tuple(entry[valid_tracks] for entry in data),
            # )
            # new_backdrops = (
            #     track_ids[~valid_tracks],
            #     tuple(entry[~valid_tracks] for entry in data),
            # )

            # self.track_memory.update(*new_tracks)
            # self.backdrop_memory.update(*new_backdrops)
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
