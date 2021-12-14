"""mmsegmentation segmentor wrapper."""
from typing import Dict, List, Optional, Tuple, Union

import torch

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
from ..heads.dense_head import (
    MMSegDecodeHead,
    MMSegDecodeHeadConfig,
    build_dense_head,
)
from ..mmdet_utils import _parse_losses, add_keyword_args
from ..mmseg_utils import load_config
from ..utils import predictions_to_scalabel
from .base import BaseSegmentor

try:
    from mmcv import Config as MMConfig
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
    (r"^decode_head\.", "decode_head.mm_decode_head."),
    (r"^auxiliary_head\.", "auxiliary_head.mm_decode_head."),
    (r"^backbone\.", "backbone.mm_backbone."),
    (r"^neck\.", "backbone.neck.mm_neck."),
]
MMSegDecodeHeads = Union[MMSegDecodeHead, torch.nn.ModuleList]


class MMEncDecSegmentorConfig(BaseModelConfig):
    """Config for mmsegmentation encoder-decoder models."""

    model_base: str
    model_kwargs: Optional[Dict[str, Union[bool, float, str, List[float]]]]
    pixel_mean: Tuple[float, float, float]
    pixel_std: Tuple[float, float, float]
    backbone_output_names: Optional[List[str]]
    weights: Optional[str]


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
        assert self.cfg.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}
        self.mm_cfg = get_mmseg_config(self.cfg)
        self.train_cfg = (
            self.mm_cfg["train_cfg"] if "train_cfg" in self.mm_cfg else None
        )
        self.test_cfg = (
            self.mm_cfg["test_cfg"] if "test_cfg" in self.mm_cfg else None
        )

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
        assert not isinstance(
            decode_cfg, list
        ), "List of decode heads currently not yet supported."
        self.decode_head: MMSegDecodeHeads = self._build_decode_heads(
            decode_cfg
        )

        if "auxiliary_head" in self.mm_cfg:
            aux_cfg = self.mm_cfg["auxiliary_head"]
            self.auxiliary_head: Optional[
                MMSegDecodeHeads
            ] = self._build_decode_heads(aux_cfg)
        else:  # pragma: no cover
            self.auxiliary_head = None

        if self.cfg.weights is not None:
            if self.cfg.weights.startswith("mmseg://"):
                self.cfg.weights = (
                    MMSEG_MODEL_PREFIX + self.cfg.weights.split("mmseg://")[-1]
                )
            load_checkpoint(self, self.cfg.weights, revise_keys=REV_KEYS)

    def _build_decode_heads(
        self,
        mm_cfg: MMConfig,
    ) -> MMSegDecodeHeads:
        """Build decode heads given config."""
        decode_head: MMSegDecodeHeads
        if isinstance(mm_cfg, list):
            decode_head_: List[MMSegDecodeHead] = []
            for mm_cfg_ in mm_cfg:
                mm_cfg_.update(
                    train_cfg=self.train_cfg, test_cfg=self.test_cfg
                )
                decode_head_.append(
                    build_dense_head(
                        MMSegDecodeHeadConfig(
                            type="MMSegDecodeHead",
                            mm_cfg=mm_cfg_,
                            category_mapping=self.cfg.category_mapping,
                        )
                    )
                )
            decode_head = torch.nn.ModuleList(decode_head_)
        else:
            mm_cfg.update(train_cfg=self.train_cfg, test_cfg=self.test_cfg)
            decode_head = build_dense_head(
                MMSegDecodeHeadConfig(
                    type="MMSegDecodeHead",
                    mm_cfg=mm_cfg,
                    category_mapping=self.cfg.category_mapping,
                )
            )
        return decode_head

    def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
        """Forward pass during training stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in MMEncDecSegmentor training!"
        inputs, targets = batch_inputs[0], batch_inputs[0].targets
        assert targets is not None, "Training requires targets."
        features = self.extract_features(inputs)
        losses, _ = self.generate_segmentations(inputs, features, targets)
        return losses

    def forward_test(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Forward pass during testing stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in MMEncDecSegmentor testing!"
        inputs = batch_inputs[0]
        features = self.extract_features(inputs)
        segmentations = self.generate_segmentations(inputs, features)
        assert segmentations is not None

        return predictions_to_scalabel(
            inputs, dict(sem_seg=segmentations), self.cat_mapping
        )

    def extract_features(self, inputs: InputSample) -> FeatureMaps:
        """Segmentor feature extraction stage.

        Return backbone output features.
        """
        return self.backbone(inputs)

    def _segmentations_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[List[SemanticMasks]]]:
        """Train stage segmentations generation."""
        assert not isinstance(self.decode_head, torch.nn.ModuleList)
        decode_losses, _ = self.decode_head(inputs, features, targets)
        segment_losses = _parse_losses(decode_losses, "decode")
        if self.auxiliary_head is not None:
            aux_losses = self.generate_auxiliaries(inputs, features)
            segment_losses.update(aux_losses)
        return segment_losses, None

    def _segmentations_test(
        self,
        inputs: InputSample,
        features: FeatureMaps,
    ) -> List[SemanticMasks]:
        """Test stage segmentations generation."""
        assert not isinstance(self.decode_head, torch.nn.ModuleList)
        return self.decode_head(inputs, features)

    def generate_auxiliaries(
        self,
        inputs: InputSample,
        features: FeatureMaps,
    ) -> LossesType:
        """Segmentor auxiliary head stage.

        Return auxiliary losses (empty if no targets).
        """
        aux_losses = {}
        if self.auxiliary_head is not None:
            if isinstance(self.auxiliary_head, torch.nn.ModuleList):
                for idx, aux_head in enumerate(self.auxiliary_head):
                    loss_aux, _ = aux_head(inputs, features, inputs.targets)
                    aux_losses.update(_parse_losses(loss_aux, f"aux_{idx}"))
            else:
                loss_aux, _ = self.auxiliary_head(
                    inputs, features, inputs.targets
                )
                aux_losses.update(_parse_losses(loss_aux, "aux"))
        return aux_losses


def get_mmseg_config(config: MMEncDecSegmentorConfig) -> MMConfig:
    """Convert a Segmentor config to a mmseg readable config."""
    cfg = load_config(config.model_base)

    # convert segmentor attributes
    assert config.category_mapping is not None
    if isinstance(cfg["decode_head"], list):  # pragma: no cover
        for dec_head in cfg["decode_head"]:
            dec_head["num_classes"] = len(config.category_mapping)
    else:
        cfg["decode_head"]["num_classes"] = len(config.category_mapping)
    if "auxiliary_head" in cfg:
        if isinstance(cfg["auxiliary_head"], list):
            for aux_head in cfg["auxiliary_head"]:
                aux_head["num_classes"] = len(config.category_mapping)
        else:
            cfg["auxiliary_head"]["num_classes"] = len(config.category_mapping)

    if config.model_kwargs:
        add_keyword_args(config, cfg)
    return cfg
