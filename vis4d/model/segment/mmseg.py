"""mmsegmentation segmentor wrapper."""
from typing import List, Optional, Tuple

from vis4d.struct import (
    FeatureMaps,
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
    SemanticMasks,
)

from ..backbone import MMSegBackboneConfig, build_backbone
from ..backbone.neck import MMDetNeckConfig
from ..base import BaseModelConfig
from ..heads.dense_head import MMSegDecodeHeadConfig, build_dense_head
from ..mmseg_utils import MMEncDecSegmentorConfig, get_mmseg_config
from .base import BaseSegmentor

try:
    from mmcv.runner.checkpoint import load_checkpoint

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

try:
    MMSEG_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMSEG_INSTALLED = False


MMSEG_MODEL_PREFIX = "https://download.openmmlab.com/mmsegmentation/v0.5/"
REV_KEYS = [
    ("decode_head.", "decode_head.mm_decode_head."),
    ("auxiliary_head.", "auxiliary_head.mm_decode_head."),
    ("backbone.", "backbone.mm_backbone."),
]


class MMEncDecSegmentor(BaseSegmentor):
    """mmsegmentation encoder-decoder segmentor wrapper."""

    def __init__(self, cfg: BaseModelConfig):
        """Init."""
        assert (
            MMSEG_INSTALLED and MMCV_INSTALLED
        ), "MMEncDecSegmentor requires both mmcv and mmseg to be installed!"
        super().__init__(cfg)
        self.cfg: MMEncDecSegmentorConfig = MMEncDecSegmentorConfig(
            **cfg.dict()
        )
        self.mm_cfg = get_mmseg_config(self.cfg)
        self.backbone = build_backbone(
            MMSegBackboneConfig(
                type="MMSegBackbone",
                mm_cfg=self.mm_cfg["backbone"],
                pixel_mean=self.cfg.pixel_mean,
                pixel_std=self.cfg.pixel_std,
                neck=MMDetNeckConfig(
                    type="MMDetNeck",
                    mm_cfg=self.mm_cfg["neck"],
                    output_names=self.cfg.backbone_output_names,
                )
                if "neck" in self.mm_cfg
                else None,
            )
        )

        decode_cfg = self.mm_cfg["decode_head"]
        self.decode_head = build_dense_head(
            MMSegDecodeHeadConfig(type="MMSegDecodeHead", mm_cfg=decode_cfg)
        )

        if "auxiliary_head" in self.mm_cfg:
            aux_cfg = self.mm_cfg["auxiliary_head"]
            if isinstance(aux_cfg, list):
                self.auxiliary_head = [
                    build_dense_head(
                        MMSegDecodeHeadConfig(
                            type="MMSegDecodeHead", mm_cfg=aux_cfg_
                        )
                    )
                    for aux_cfg_ in aux_cfg
                ]
            else:
                self.auxiliary_head = build_dense_head(
                    MMSegDecodeHeadConfig(
                        type="MMSegDecodeHead", mm_cfg=aux_cfg
                    )
                )
        else:
            self.auxiliary_head = None

        if self.cfg.weights is not None:
            if self.cfg.weights.startswith("mmseg://"):
                self.cfg.weights = (
                    MMSEG_MODEL_PREFIX + self.cfg.weights.split("mmseg://")[-1]
                )
            load_checkpoint(self, self.cfg.weights, revise_keys=REV_KEYS)

        assert self.cfg.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}

    def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
        """Forward pass during training stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in MMEncDecSegmentor training!"
        inputs = batch_inputs[0]
        features = self.backbone(inputs)
        decode_losses = self.decode_head(inputs, features, inputs.targets)
        aux_losses = {}
        if self.auxiliary_head is not None:
            aux_losses = self.auxiliary_head(inputs, features, inputs.targets)
        return {**decode_losses, **aux_losses}

    def forward_test(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Forward pass during testing stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in MMEncDecSegmentor testing!"
        inputs = batch_inputs[0]
        features = self.backbone(inputs)
        segmentations = self.decode_head(inputs, features)
        assert segmentations is not None

        return dict(
            sem_seg=[s.to_scalabel(self.cat_mapping) for s in segmentations]
        )

    def extract_features(self, inputs: InputSample) -> FeatureMaps:
        """Segmentor feature extraction stage.

        Return backbone output features.
        """
        return self.backbone(inputs)

    def generate_segmentations(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: Optional[LabelInstances] = None,
    ) -> Tuple[LossesType, Optional[List[SemanticMasks]]]:  # pragma: no cover
        """Segmentor decode stage.

        Return losses (empty if not training) and optionally segmentations.
        """
        decode_output = self.decode_head(inputs, features, inputs.targets)
        if self.training:
            segment_losses = decode_output
            if self.auxiliary_head is not None:
                aux_losses = self.generate_auxiliaries(inputs, features)
                segment_losses.update(aux_losses)
            segmentations = None
        else:
            segmentations = decode_output
            segment_losses = {}
        return segment_losses, segmentations

    def generate_auxiliaries(
        self,
        inputs: InputSample,
        features: FeatureMaps,
    ) -> LossesType:  # pragma: no cover
        """Segmentor auxiliary head stage.

        Return auxiliary losses (empty if no targets).
        """
        return self.auxiliary_head(inputs, features, inputs.targets)
