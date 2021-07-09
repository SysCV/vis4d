"""mmdetection detector wrapper."""
from typing import Dict, List, Optional, Tuple

import torch
from mmcv.runner.checkpoint import load_checkpoint
from mmdet.models import TwoStageDetector, build_detector

from openmt.struct import Boxes2D, Images, InputSample, LossesType, ModelOutput

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


class MMTwoStageDetector(BaseTwoStageDetector):
    """mmdetection two-stage detector wrapper."""

    def __init__(self, cfg: BaseModelConfig):
        """Init."""
        super().__init__()
        self.cfg = MMTwoStageDetectorConfig(**cfg.dict())
        self.mm_cfg = get_mmdet_config(self.cfg)
        self.mm_detector = build_detector(self.mm_cfg)
        assert isinstance(self.mm_detector, TwoStageDetector)
        self.mm_detector.init_weights()
        self.mm_detector.train()
        if self.cfg.weights is not None:
            load_checkpoint(self.mm_detector, self.cfg.weights)

        self.register_buffer(
            "pixel_mean",
            torch.tensor(self.cfg.pixel_mean).view(-1, 1, 1),
            False,
        )
        self.register_buffer(
            "pixel_std", torch.tensor(self.cfg.pixel_std).view(-1, 1, 1), False
        )

    @property
    def device(self) -> torch.device:
        """Get device where detect input should be moved to."""
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs: List[InputSample]) -> Images:
        """Batch, pad (standard stride=32) and normalize the input images."""
        images = Images.cat([inp.image for inp in batched_inputs])
        images.tensor = images.tensor.contiguous()
        images = images.to(self.device)
        images.tensor = (images.tensor - self.pixel_mean) / self.pixel_std
        return images

    def forward_train(
        self,
        batch_inputs: List[List[InputSample]],
    ) -> LossesType:
        """Forward pass during training stage.

        Returns a dict of loss tensors.
        """
        assert all(
            len(inp) == 1 for inp in batch_inputs
        ), "No reference views allowed in detector training!"
        inputs = [inp[0] for inp in batch_inputs]
        targets = [
            x.instances.to(self.device) for x in inputs  # type: ignore
        ]  # type: List[Boxes2D]

        # from openmt.vis.image import imshow_bboxes
        # for inp in inputs:
        #     imshow_bboxes(inp.image.tensor[0], inp.instances, mode="RGB")

        images = self.preprocess_image(inputs)
        image_metas = get_img_metas(images)
        gt_bboxes, gt_labels = targets_to_mmdet(targets)
        losses = self.mm_detector.forward_train(
            images.tensor,
            image_metas,
            gt_bboxes,
            gt_labels,
        )
        return _parse_losses(losses)

    def forward_test(
        self, batch_inputs: List[InputSample], postprocess: bool = True
    ) -> ModelOutput:
        """Forward pass during testing stage.

        Returns predictions for each input.
        """
        images = self.preprocess_image(batch_inputs)
        image_metas = get_img_metas(images)
        outs = self.mm_detector.simple_test(images.tensor, image_metas)
        detections = results_from_mmdet(outs, self.device)

        if postprocess:
            for inp, det in zip(batch_inputs, detections):
                ori_wh = (
                    batch_inputs[0].metadata.size.width,  # type: ignore
                    batch_inputs[0].metadata.size.height,  # type: ignore
                )
                self.postprocess(ori_wh, inp.image.image_sizes[0], det)

        return dict(detect=detections)  # type: ignore

    def extract_features(self, images: Images) -> Dict[str, torch.Tensor]:
        """Detector feature extraction stage.

        Return preprocessed images, backbone output features.
        """
        outs = self.mm_detector.extract_feat(images.tensor)
        if self.cfg.backbone_output_names is None:  # pragma: no cover
            return {f"out{i}": v for i, v in enumerate(outs)}

        return dict(zip(self.cfg.backbone_output_names, outs))

    def generate_proposals(
        self,
        images: Images,
        features: Dict[str, torch.Tensor],
        targets: Optional[List[Boxes2D]] = None,
    ) -> Tuple[List[Boxes2D], LossesType]:
        """Detector RPN stage.

        Return proposals per image and losses (empty if no targets).
        """
        feat_list = list(features.values())
        img_metas = get_img_metas(images)
        if targets is not None:
            gt_bboxes, _ = targets_to_mmdet(targets)

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
            rpn_losses = dict()

        return proposals_from_mmdet(proposals), _parse_losses(rpn_losses)

    def generate_detections(
        self,
        images: Images,
        features: Dict[str, torch.Tensor],
        proposals: List[Boxes2D],
        targets: Optional[List[Boxes2D]] = None,
        compute_detections: bool = True,
    ) -> Tuple[Optional[List[Boxes2D]], LossesType]:
        """Detector second stage (RoI Head).

        Return losses (empty if no targets) and optionally detections.
        """
        proposal_list = proposals_to_mmdet(proposals)
        feat_list = list(features.values())
        img_metas = get_img_metas(images)
        if targets is not None:
            gt_bboxes, gt_labels = targets_to_mmdet(targets)
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
            detect_losses = dict()

        return detections, detect_losses
