"""Quasi-dense instance similarity learning model."""
import pickle
from typing import Dict, List, Optional, Tuple, Union

import torch
from pytorch_lightning.utilities.cli import instantiate_class
from torch import nn

from vis4d.common.data_pipelines import default as default_augs
from vis4d.model.detect import FasterRCNN
from vis4d.model.heads.roi_head.rcnn import TransformRCNNOutputs
from vis4d.model.optimize import DefaultOptimizer
from vis4d.model.track.graph import QDTrackGraph
from vis4d.model.track.similarity import QDSimilarityHead
from vis4d.struct import ArgsType, Tracks

try:
    from mmdet.core.bbox.assigners import SimOTAAssigner

    MMDET_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMDET_INSTALLED = False


class QDTrack(nn.Module):
    """QDTrack model - quasi-dense instance similarity learning."""

    def __init__(
        self, detector: FasterRCNN, detector_transform: TransformRCNNOutputs
    ) -> None:
        """Init."""
        super().__init__()
        self.detector = detector
        self.detector_transform = detector_transform
        self.similarity_head = QDSimilarityHead()
        self.track_graph = QDTrackGraph()

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
        self, images: torch.Tensor, frame_ids: Tuple[int, ...]
    ) -> List[Tracks]:
        """Forward function for training."""
        # detection
        detector_out = self.detector(images)
        detections = self.detector_transform(
            detector_out.roi_cls_out,
            detector_out.roi_reg_out,
            detector_out.proposal_boxes,
            images.shape,
        )
        boxes, scores, class_ids = (
            [d.boxes for d in detections],
            [d.scores for d in detections],
            [d.class_ids for d in detections],
        )

        # similarity head
        embeddings = self.similarity_head(detector_out.backbone_out[:4], boxes)

        # track graph
        batched_tracks = []
        for frame_id, box, score, cls_id, embeds in zip(
            frame_ids, boxes, scores, class_ids, embeddings
        ):
            tracks = self.track_graph(box, score, cls_id, embeds, frame_id)
            batched_tracks.append(tracks)
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
