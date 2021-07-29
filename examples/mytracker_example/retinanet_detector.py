"""Example for dynamic api usage."""
from typing import List, Dict, Optional
import torch

from detectron2.modeling.meta_arch.retinanet import RetinaNet
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Instances, ImageList
from detectron2 import model_zoo
from detectron2.config import get_cfg

from openmt.model.detect.d2_utils import (
    images_to_imagelist,
    target_to_instance,
)

from openmt.model.detect import BaseDetector
from openmt.model import BaseModelConfig
from openmt.struct import Boxes2D, Images, InputSample, LossesType, ModelOutput


def permute_to_N_HWA_K(tensor, K: int):
    """Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)."""
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def detections_to_box2d(detections: List[Instances]) -> List[Boxes2D]:
    """Convert d2 Instances representing detections to Boxes2D.

    with class_ids
    """
    result = []
    for detection in detections:
        boxes, scores, cls = (
            detection.pred_boxes.tensor,
            detection.scores,
            detection.pred_classes,
        )
        result.append(
            Boxes2D(
                torch.cat([boxes, scores.unsqueeze(-1)], -1),
                class_ids=cls,
            )
        )
    return result


class D2RetinaNetConfig(BaseModelConfig, extra="allow"):
    """Config for detectron2 RetinaNet model."""

    num_classes: int


class D2RetinaNetDetector(BaseDetector):
    """Example detection module."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init detector."""
        super().__init__()
        self.cfg = D2RetinaNetConfig(**cfg.dict())
        self.d2_cfg = get_cfg()
        base_cfg = model_zoo.get_config_file(
            "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
        )
        self.d2_cfg.merge_from_file(base_cfg)
        self.d2_cfg.MODEL.RETINANET.NUM_CLASSES = self.cfg.num_classes
        # pylint: disable=too-many-function-args,missing-kwoa
        self.retinanet = RetinaNet(self.d2_cfg)
        self.count = 0

    @property
    def device(self) -> torch.device:
        """Get device where detect input should be moved to."""
        return self.retinanet.pixel_mean.device

    def preprocess_image(self, batched_inputs: List[InputSample]) -> Images:
        """Normalize, pad and batch the input images."""
        # images = []
        # for inp in batched_inputs:
        #     t = torch.div(inp.image.tensor, 255.0)
        #     images.append(Images(t, inp.image.image_sizes))
        # images = Images.cat(images)
        images = Images.cat([inp.image for inp in batched_inputs])
        images = images.to(self.device)
        images.tensor = (
            images.tensor - self.retinanet.pixel_mean
        ) / self.retinanet.pixel_std
        return images

    def forward_train(
        self, batch_inputs: List[List[InputSample]]
    ) -> LossesType:
        """Forward pass during training stage.

        Returns a dict of loss tensors.
        """
        inputs = [inp[0] for inp in batch_inputs]  # no ref views
        targets = [x.instances.to(self.device) for x in inputs]
        images = self.preprocess_image(inputs)
        images_d2 = images_to_imagelist(images)
        targets_d2 = target_to_instance(targets, images.image_sizes)  # type: ignore

        features = self.extract_features(images_d2)

        anchors = self.retinanet.anchor_generator(features)
        pred_logits, pred_anchor_deltas = self.retinanet.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [
            permute_to_N_HWA_K(x, self.retinanet.num_classes)
            for x in pred_logits
        ]
        pred_anchor_deltas = [
            permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas
        ]
        gt_labels, gt_boxes = self.retinanet.label_anchors(anchors, targets_d2)
        detect_losses = self.retinanet.losses(
            anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes
        )
        return detect_losses  # type: ignore

    def forward_test(
        self, batch_inputs: List[InputSample], postprocess: bool = True
    ) -> ModelOutput:
        """Forward pass during testing stage.

        Returns predictions for each input.
        """
        images = self.preprocess_image(batch_inputs)
        images_d2 = images_to_imagelist(images)
        features = self.extract_features(images_d2)

        anchors = self.retinanet.anchor_generator(features)
        pred_logits, pred_anchor_deltas = self.retinanet.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [
            permute_to_N_HWA_K(x, self.retinanet.num_classes)
            for x in pred_logits
        ]
        pred_anchor_deltas = [
            permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas
        ]

        results = self.retinanet.inference(
            anchors, pred_logits, pred_anchor_deltas, images_d2.image_sizes
        )

        processed_results = []
        for results_per_image, image_size in zip(
            results, images_d2.image_sizes
        ):
            height = image_size[0]
            width = image_size[1]
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append(r)

        detections = detections_to_box2d(processed_results)

        for inp, det in zip(batch_inputs, detections):
            ori_wh = (
                batch_inputs[0].metadata.size.width,  # type: ignore
                batch_inputs[0].metadata.size.height,  # type: ignore
            )
            self.postprocess(ori_wh, inp.image.image_sizes[0], det)

        return dict(detect=detections)  # type:ignore

    def extract_features(
        self, images: ImageList
    ) -> List[torch.Tensor]:  # type:ignore
        """Detector feature extraction stage.

        Return backbone output features
        """

        # backbone
        features = self.retinanet.backbone(images.tensor)
        features = [features[f] for f in self.retinanet.head_in_features]
        return features

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
        raise NotImplementedError
