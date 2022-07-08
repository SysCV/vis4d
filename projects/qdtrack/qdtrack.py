"""Quasi-dense instance similarity learning model."""
from typing import Tuple

import torch
from pytorch_lightning.utilities.cli import instantiate_class

from projects.common.data_pipelines import default as default_augs
from vis4d.model import QDTrack
from vis4d.model.optimize import DefaultOptimizer
from vis4d.struct import ArgsType


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
        assert MMDET_AVAILABLE, "QDTrackYOLOX needs mmdet installed!"
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
