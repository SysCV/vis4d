"""mmsegmentation segmentor wrapper."""
from typing import Dict, List, Optional, Tuple, Union

import torch

from vis4d.struct import (
    ArgsType,
    DictStrAny,
    FeatureMaps,
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
    SemanticMasks,
)

from ..backbone import MMSegBackbone
from ..backbone.neck import MMDetNeck
from ..heads.dense_head import MMSegDecodeHead
from ..mmdet_utils import _parse_losses, add_keyword_args
from ..mmseg_utils import load_config
from ..utils import postprocess_predictions, predictions_to_scalabel
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
BDD100K_MODEL_PREFIX = "https://dl.cv.ethz.ch/bdd100k/"
REV_KEYS = [
    (r"^decode_head\.", "decode_head.mm_decode_head."),
    (r"^auxiliary_head\.", "auxiliary_head.mm_decode_head."),
    (r"^backbone\.", "backbone.mm_backbone."),
    (r"^neck\.", "backbone.neck.mm_neck."),
]
MMSegDecodeHeads = Union[MMSegDecodeHead, torch.nn.ModuleList]


class MMEncDecSegmentor(BaseSegmentor):
    """mmsegmentation encoder-decoder segmentor wrapper."""

    def __init__(
        self,
        model_base: str,
        pixel_mean: Tuple[float, float, float],
        pixel_std: Tuple[float, float, float],
        *args: ArgsType,
        model_kwargs: Optional[DictStrAny] = None,
        backbone_output_names: Optional[List[str]] = None,
        weights: Optional[str] = None,
        **kwargs: ArgsType,
    ):
        """Init."""
        assert (
            MMSEG_INSTALLED and MMCV_INSTALLED
        ), "MMEncDecSegmentor requires both mmcv and mmseg to be installed!"
        super().__init__(*args, **kwargs)
        assert self.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.category_mapping.items()}
        self.mm_cfg = get_mmseg_config(
            model_base, model_kwargs, self.category_mapping
        )
        self.train_cfg = (
            self.mm_cfg["train_cfg"] if "train_cfg" in self.mm_cfg else None
        )
        self.test_cfg = (
            self.mm_cfg["test_cfg"] if "test_cfg" in self.mm_cfg else None
        )

        self.backbone = MMSegBackbone(
            mm_cfg=self.mm_cfg["backbone"],
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            neck=MMDetNeck(
                mm_cfg=self.mm_cfg["neck"],
                output_names=backbone_output_names,
            )
            if "neck" in self.mm_cfg
            else None,
            output_names=backbone_output_names,
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

        if weights is not None:
            load_model_checkpoint(self, weights)

    def _build_decode_heads(
        self,
        mm_cfg: MMConfig,
    ) -> MMSegDecodeHeads:
        """Build decode heads given config."""
        assert self.category_mapping is not None
        decode_head: MMSegDecodeHeads
        if isinstance(mm_cfg, list):
            decode_head_: List[MMSegDecodeHead] = []
            for mm_cfg_ in mm_cfg:
                mm_cfg_.update(
                    train_cfg=self.train_cfg, test_cfg=self.test_cfg
                )
                decode_head_.append(
                    MMSegDecodeHead(
                        mm_cfg=mm_cfg_, category_mapping=self.category_mapping
                    )
                )
            decode_head = torch.nn.ModuleList(decode_head_)
        else:
            mm_cfg.update(train_cfg=self.train_cfg, test_cfg=self.test_cfg)
            decode_head = MMSegDecodeHead(
                mm_cfg=mm_cfg, category_mapping=self.category_mapping
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

        outputs = dict(sem_seg=segmentations)
        postprocess_predictions(inputs, outputs)
        return predictions_to_scalabel(outputs, self.cat_mapping)

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


def load_model_checkpoint(model: BaseSegmentor, weights: str) -> None:
    """Load MMSeg model checkpoint."""
    if weights.startswith("mmseg://"):
        weights = MMSEG_MODEL_PREFIX + weights.split("mmseg://")[-1]
    elif weights.startswith("bdd100k://"):  # pragma: no cover
        weights = BDD100K_MODEL_PREFIX + weights.split("bdd100k://")[-1]
    load_checkpoint(model, weights, revise_keys=REV_KEYS)


def get_mmseg_config(
    model_base: str,
    model_kwargs: Optional[DictStrAny] = None,
    category_mapping: Optional[Dict[str, int]] = None,
) -> MMConfig:
    """Convert a Segmentor config to a mmseg readable config."""
    cfg = load_config(model_base)

    # convert segmentor attributes
    assert category_mapping is not None
    if isinstance(cfg["decode_head"], list):  # pragma: no cover
        for dec_head in cfg["decode_head"]:
            dec_head["num_classes"] = len(category_mapping)
    else:
        cfg["decode_head"]["num_classes"] = len(category_mapping)
    if "auxiliary_head" in cfg:
        if isinstance(cfg["auxiliary_head"], list):
            for aux_head in cfg["auxiliary_head"]:
                aux_head["num_classes"] = len(category_mapping)
        else:
            cfg["auxiliary_head"]["num_classes"] = len(category_mapping)

    if model_kwargs:
        add_keyword_args(model_kwargs, cfg)
    return cfg
