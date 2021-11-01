"""mmsegmentation segmentor wrapper."""
from typing import Dict, List, Optional, Tuple

import torch
from mmcv.runner.checkpoint import load_checkpoint
from mmseg.models import EncoderDecoder, build_segmentor

from vist.struct import InputSample, LossesType, ModelOutput, SemanticMasks

from ..base import BaseModelConfig
from ..detect.mmdet_utils import _parse_losses, get_img_metas
from .base import BaseSegmentor
from .mmseg_utils import (
    MMEncDecSegmentorConfig,
    get_mmseg_config,
    results_from_mmseg,
    segmentations_from_mmseg,
    targets_to_mmseg,
)

MMSEG_MODEL_PREFIX = "https://download.openmmlab.com/mmsegmentation/v0.5/"


class MMEncDecSegmentor(BaseSegmentor):
    """mmsegmentation encoder-decoder segmentor wrapper."""

    def __init__(self, cfg: BaseModelConfig):
        """Init."""
        super().__init__(cfg)
        self.cfg = MMEncDecSegmentorConfig(
            **cfg.dict()
        )  # type: MMEncDecSegmentorConfig
        self.mm_cfg = get_mmseg_config(self.cfg)
        self.mm_segmentor = build_segmentor(self.mm_cfg)
        assert isinstance(self.mm_segmentor, EncoderDecoder)
        self.mm_segmentor.init_weights()
        self.mm_segmentor.train()
        if self.cfg.weights is not None:
            if self.cfg.weights.startswith("mmseg://"):
                self.cfg.weights = (
                    MMSEG_MODEL_PREFIX + self.cfg.weights.split("mmseg://")[-1]
                )
            load_checkpoint(self.mm_segmentor, self.cfg.weights)

        assert self.cfg.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}
        self.register_buffer(
            "pixel_mean",
            torch.tensor(self.cfg.pixel_mean).view(-1, 1, 1),
            False,
        )
        self.register_buffer(
            "pixel_std", torch.tensor(self.cfg.pixel_std).view(-1, 1, 1), False
        )

    def preprocess_inputs(self, inputs: List[InputSample]) -> InputSample:
        """Batch, pad (standard stride=32) and normalize the input images."""
        batched_inputs = InputSample.cat(inputs, self.device)
        batched_inputs.images.tensor = (
            batched_inputs.images.tensor - self.pixel_mean
        ) / self.pixel_std
        return batched_inputs

    def forward_train(
        self, batch_inputs: List[List[InputSample]]
    ) -> LossesType:
        """Forward pass during training stage."""
        assert all(
            len(inp) == 1 for inp in batch_inputs
        ), "No reference views allowed in MMEncDecSegmentor training!"
        raw_inputs = [inp[0] for inp in batch_inputs]
        inputs = self.preprocess_inputs(raw_inputs)

        image_metas = get_img_metas(inputs.images)
        gt_masks = targets_to_mmseg(inputs)
        losses = self.mm_segmentor.forward_train(
            inputs.images.tensor, image_metas, gt_masks
        )
        return _parse_losses(losses)

    def forward_test(
        self,
        batch_inputs: List[List[InputSample]],
    ) -> ModelOutput:
        """Forward pass during testing stage."""
        raw_inputs = [inp[0] for inp in batch_inputs]
        inputs = self.preprocess_inputs(raw_inputs)
        image_metas = get_img_metas(inputs.images)
        outs = self.mm_segmentor.simple_test(inputs.images.tensor, image_metas)
        segmentations = results_from_mmseg(outs, self.device)
        assert segmentations is not None

        return dict(
            sem_seg=[s.to_scalabel(self.cat_mapping) for s in segmentations]
        )

    def extract_features(
        self, inputs: InputSample
    ) -> Dict[str, torch.Tensor]:  # pragma: no cover
        """Segmentor feature extraction stage.

        Return backbone output features
        """
        outs = self.mm_segmentor.extract_feat(inputs.images.tensor)
        if self.cfg.backbone_output_names is None:
            return {f"out{i}": v for i, v in enumerate(outs)}

        return dict(zip(self.cfg.backbone_output_names, outs))

    def generate_segmentations(
        self,
        inputs: InputSample,
        features: Dict[str, torch.Tensor],
        compute_segmentations: bool = True,
    ) -> Tuple[Optional[List[SemanticMasks]], LossesType]:  # pragma: no cover
        """Segmentor decode stage.

        Return losses (empty if not training) and optionally segmentations.
        """
        feat_list = list(features.values())
        img_metas = get_img_metas(inputs.images)
        if self.training:
            gt_masks = targets_to_mmseg(inputs)
            segment_losses = self.mm_segmentor.decode_head.forward_train(
                feat_list, img_metas, gt_masks, self.mm_segmentor.train_cfg
            )
            segment_losses = _parse_losses(segment_losses, "decode")
            assert (
                not compute_segmentations
            ), "mmsegmentation does not compute segmentations during train!"
            segmentations = None
        else:
            masks = self.mm_segmentor.decode_head.forward_test(
                feat_list, img_metas, self.mm_segmentor.test_cfg
            )
            segmentations = segmentations_from_mmseg(masks, self.device)
            segment_losses = {}

        return segmentations, segment_losses

    def generate_auxiliaries(
        self,
        inputs: InputSample,
        features: Dict[str, torch.Tensor],
    ) -> LossesType:  # pragma: no cover
        """Segmentor auxiliary head stage.

        Return auxiliary losses (empty if no targets).
        """
        aux_losses = {}
        if self.training:
            feat_list = list(features.values())
            img_metas = get_img_metas(inputs.images)
            gt_masks = targets_to_mmseg(inputs)
            if isinstance(
                self.mm_segmentor.auxiliary_head, torch.nn.ModuleList
            ):
                for idx, aux_head in enumerate(
                    self.mm_segmentor.auxiliary_head
                ):
                    loss_aux = aux_head.forward_train(
                        feat_list,
                        img_metas,
                        gt_masks,
                        self.mm_segmentor.train_cfg,
                    )
                    aux_losses.update(_parse_losses(loss_aux, f"aux_{idx}"))
            else:
                loss_aux = self.mm_segmentor.auxiliary_head.forward_train(
                    feat_list, img_metas, gt_masks, self.mm_segmentor.train_cfg
                )
                aux_losses.update(_parse_losses(loss_aux, "aux"))

        return aux_losses
