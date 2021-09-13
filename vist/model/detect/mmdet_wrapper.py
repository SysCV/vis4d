"""mmdetection detector wrapper."""
from typing import Dict, List, Optional, Tuple

import torch
from mmcv.runner.checkpoint import load_checkpoint
from mmdet.models import TwoStageDetector, build_detector

from vist.struct import Boxes2D, InputSample, LossesType, ModelOutput

from ..base import BaseModelConfig
from .base import BaseTwoStageDetector
from .mmdet_utils import (
    MMTwoStageDetectorConfig,
    _parse_losses,
    detections_from_mmdet,
    get_img_metas,
    get_mmdet_config,
    proposals_from_mmdet,
    proposals_to_mmdet,
    results_from_mmdet,
    targets_to_mmdet,
)

MMDET_MODEL_PREFIX = "https://download.openmmlab.com/mmdetection/v2.0/"


class MMTwoStageDetector(BaseTwoStageDetector):
    """mmdetection two-stage detector wrapper."""

    def __init__(self, cfg: BaseModelConfig):
        """Init."""
        super().__init__(cfg)
        self.cfg = MMTwoStageDetectorConfig(
            **cfg.dict()
        )  # type: MMTwoStageDetectorConfig
        self.mm_cfg = get_mmdet_config(self.cfg)
        self.mm_detector = build_detector(self.mm_cfg)
        assert isinstance(self.mm_detector, TwoStageDetector)
        self.mm_detector.init_weights()
        self.mm_detector.train()
        if self.cfg.weights is not None:
            if self.cfg.weights.startswith("mmdet://"):
                self.cfg.weights = MMDET_MODEL_PREFIX + self.cfg.weights.strip(
                    "mmdet://"
                )
            load_checkpoint(self.mm_detector, self.cfg.weights)

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
        self,
        batch_inputs: List[List[InputSample]],
    ) -> LossesType:
        """Forward pass during training stage."""
        assert all(
            len(inp) == 1 for inp in batch_inputs
        ), "No reference views allowed in MMTwoStageDetector training!"
        raw_inputs = [inp[0] for inp in batch_inputs]
        inputs = self.preprocess_inputs(raw_inputs)
        image_metas = get_img_metas(inputs.images)
        gt_bboxes, gt_labels = targets_to_mmdet(inputs.boxes2d)
        losses = self.mm_detector.forward_train(
            inputs.images.tensor,
            image_metas,
            gt_bboxes,
            gt_labels,
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
        outs = self.mm_detector.simple_test(inputs.images.tensor, image_metas)
        detections = results_from_mmdet(outs, self.device)
        assert detections is not None

        for inp, det in zip(inputs, detections):
            assert inp.metadata[0].size is not None
            input_size = (
                inp.metadata[0].size.width,
                inp.metadata[0].size.height,
            )
            self.postprocess(input_size, inp.images.image_sizes[0], det)

        return dict(detect=detections)  # type: ignore

    def extract_features(self, inputs: InputSample) -> Dict[str, torch.Tensor]:
        """Detector feature extraction stage.

        Return backbone output features.
        """
        outs = self.mm_detector.extract_feat(inputs.images.tensor)
        if self.cfg.backbone_output_names is None:  # pragma: no cover
            return {f"out{i}": v for i, v in enumerate(outs)}

        return dict(zip(self.cfg.backbone_output_names, outs))

    def generate_proposals(
        self,
        inputs: InputSample,
        features: Dict[str, torch.Tensor],
    ) -> Tuple[List[Boxes2D], LossesType]:
        """Detector RPN stage.

        Return proposals per image and losses (empty if no targets).
        """
        feat_list = list(features.values())
        img_metas = get_img_metas(inputs.images)
        if self.training:
            gt_bboxes, _ = targets_to_mmdet(inputs.boxes2d)

            proposal_cfg = self.mm_detector.train_cfg.get(
                "rpn_proposal", self.mm_detector.test_cfg.rpn
            )
            rpn_losses, proposals = self.mm_detector.rpn_head.forward_train(
                feat_list,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                proposal_cfg=proposal_cfg,
            )
        else:
            proposals = self.mm_detector.rpn_head.simple_test(
                feat_list, img_metas
            )
            rpn_losses = {}

        return proposals_from_mmdet(proposals), _parse_losses(rpn_losses)

    def generate_detections(
        self,
        inputs: InputSample,
        features: Dict[str, torch.Tensor],
        proposals: Optional[List[Boxes2D]] = None,
        compute_detections: bool = True,
    ) -> Tuple[Optional[List[Boxes2D]], LossesType]:
        """Detector second stage (RoI Head).

        Return losses (empty if no targets) and optionally detections.
        """
        assert (
            proposals is not None
        ), "Generating detections with MMTwoStageDetector requires proposals."
        proposal_list = proposals_to_mmdet(proposals)
        feat_list = list(features.values())
        img_metas = get_img_metas(inputs.images)
        if self.training:
            gt_bboxes, gt_labels = targets_to_mmdet(inputs.boxes2d)
            detect_losses = self.mm_detector.roi_head.forward_train(
                feat_list,
                img_metas,
                proposal_list,
                gt_bboxes,
                gt_labels,
            )
            detect_losses = _parse_losses(detect_losses)
            assert (
                not compute_detections
            ), "mmdetection does not compute detections during train!"
            detections = None
        else:
            bboxes, labels = self.mm_detector.roi_head.simple_test_bboxes(
                feat_list,
                img_metas,
                proposal_list,
                self.mm_detector.roi_head.test_cfg,
            )
            detections = detections_from_mmdet(bboxes, labels)
            detect_losses = {}

        return detections, detect_losses
