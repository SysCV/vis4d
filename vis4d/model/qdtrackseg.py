"""Quasi-dense instance similarity learning model with segmentation head."""
from typing import List

import torch

from vis4d.struct import InputSample, LossesType, ModelOutput

from .base import BaseModelConfig
from .heads.dense_head import (
    BaseDenseHeadConfig,
    MMSegDecodeHead,
    build_dense_head,
)
from .qdtrack import QDTrack, QDTrackConfig
from .track.utils import split_key_ref_inputs


class QDTrackSegConfig(QDTrackConfig):
    """Config for quasi-dense tracking model with segmentation head."""

    seg_head: BaseDenseHeadConfig


class QDTrackSeg(QDTrack):
    """QDTrack model with segmentation head."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg: QDTrackSegConfig = QDTrackSegConfig(**cfg.dict())
        if self.cfg.seg_head.category_mapping is None:  # pragma: no cover
            self.cfg.seg_head.category_mapping = self.cfg.category_mapping
        self.seg_head: MMSegDecodeHead = build_dense_head(self.cfg.seg_head)

    def forward_train(
        self,
        batch_inputs: List[InputSample],
    ) -> LossesType:
        """Forward function for training."""
        key_inputs, ref_inputs = split_key_ref_inputs(batch_inputs)
        key_targets = key_inputs.targets

        # feature extraction
        key_x = self.detector.extract_features(key_inputs)
        ref_x = [self.detector.extract_features(inp) for inp in ref_inputs]

        losses = {}
        if len(key_targets.boxes2d[0]) > 0:
            track_losses, _, _ = self._detect_and_track_losses(
                key_inputs, ref_inputs, key_x, ref_x
            )
            losses.update(track_losses)

        if (
            len(key_targets.semantic_masks[0]) > 0
            and self.seg_head is not None
        ):
            # segmentation head
            seg_losses, _ = self.seg_head(
                key_inputs, key_x, key_inputs.targets
            )
            losses.update(seg_losses)

        return losses

    def forward_test(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Compute model output during inference."""
        assert len(batch_inputs) == 1, "No reference views during test!"
        assert len(batch_inputs[0]) == 1, "Currently only BS=1 supported!"
        feat = self.detector.extract_features(batch_inputs[0])
        outputs = self._detect_and_track(batch_inputs[0], feat)

        # segmentation head
        semantic_segms = self.seg_head(batch_inputs[0], feat)
        semantic_segms_ = (
            semantic_segms[0]
            .to(torch.device("cpu"))
            .to_scalabel(self.seg_head.cat_mapping)
        )
        outputs["sem_seg"] = [semantic_segms_]

        return outputs
