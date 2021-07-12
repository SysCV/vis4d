"""Example for dynamic api usage."""
from typing import List, Dict, Union, Optional
import os
import torch

from openmt.model.detect.d2_utils import (
    images_to_imagelist,
    target_to_instance,
)
from openmt import config
from openmt.data.build import DataloaderConfig as Dataloader

# from openmt.config import DataloaderConfig as Dataloader
from openmt.engine import train
from openmt.model.detect import (
    BaseDetector,
    BaseDetectorConfig,
)
from openmt.struct import Boxes2D, DetectionOutput, Images, InputSample

from detectron2.modeling.meta_arch.retinanet import RetinaNet, RetinaNetHead
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg


def permute_to_N_HWA_K(tensor, K: int):
    """Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)"""
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def detections_to_box2d_training(detections: List[Instances]) -> List[Boxes2D]:
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


def detections_to_box2d_inference(
    detections: List[Instances],
) -> List[Boxes2D]:
    """Convert d2 Instances representing detections to Boxes2D.

    no class_ids
    """
    result = []
    for detection in detections:
        boxes, logits = (
            detection.pred_boxes.tensor,
            detection.scores,
        )
        result.append(
            Boxes2D(
                torch.cat([boxes, logits.unsqueeze(-1)], -1),
            )
        )
    return result


class D2RetinaNetConfig(BaseDetectorConfig, extra="allow"):
    """Config for detectron2 RetinaNet model."""

    num_classes: int


class D2RetinaNetDetector(BaseDetector):
    """Example detection module."""

    def __init__(self, cfg: BaseDetectorConfig) -> None:
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

    def forward(
        self,
        inputs: List[InputSample],
        targets: Optional[List[Boxes2D]] = None,
    ) -> DetectionOutput:
        """Detector forward function.

        Return backbone output features, proposals, detections and optionally
        training losses.
        """

        def to_batched_inputs(images_d2, targets):
            batched_inputs = []
            if targets is None:
                for image in images_d2:
                    batched_inputs.append(
                        {"image": image.cpu(), "instances": None}
                    )
            else:
                for image, target in zip(images_d2, targets):
                    batched_inputs.append(
                        {"image": image.cpu(), "instances": target.to("cpu")}
                    )
            return batched_inputs

        images = self.preprocess_image(inputs)
        images_d2 = images_to_imagelist(images)
        if targets is not None:
            targets_d2 = target_to_instance(targets, images.image_sizes)

        # batched_inputs = to_batched_inputs(images_d2, targets)
        # backbone
        features = self.retinanet.backbone(images_d2.tensor)
        # features = [f for k, f in features.items()]
        features = [features[f] for f in self.retinanet.head_in_features]

        anchors = self.retinanet.anchor_generator(features)

        # from openmt.vis.image import imshow_bboxes

        # boxes2d_anchors = Boxes2D(
        #     torch.cat(
        #         (
        #             anchors[1].tensor,
        #             0.7
        #             * torch.ones(anchors[1].tensor.shape[0], 1).to(
        #                 self.device
        #             ),
        #         ),
        #         dim=1,
        #     )
        # )
        # imshow_bboxes(
        #     torch.zeros(
        #         size=tuple(
        #             [3, features[1][0].shape[1], features[1][0].shape[2]]
        #         ),
        #     ),
        #     boxes2d_anchors,
        #     frame_id=self.count,
        # )
        pred_logits, pred_anchor_deltas = self.retinanet.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [
            permute_to_N_HWA_K(x, self.retinanet.num_classes)
            for x in pred_logits
        ]
        pred_anchor_deltas = [
            permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas
        ]

        if self.retinanet.training:
            gt_labels, gt_boxes = self.retinanet.label_anchors(
                anchors, targets_d2
            )
            detect_losses = self.retinanet.losses(
                anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
        else:
            detect_losses = {}
        results = self.retinanet.inference(
            anchors, pred_logits, pred_anchor_deltas, images_d2.image_sizes
        )
        # vil_results = [results[0].to("cpu")]
        # if self.count % 100 == 0:
        #     self.retinanet.visualize_training(batched_inputs, vil_results)
        processed_results = []
        for results_per_image, image_size in zip(
            results, images_d2.image_sizes
        ):
            height = image_size[0]
            width = image_size[1]
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append(r)

        if not self.retinanet.training:
            detections = detections_to_box2d_training(processed_results)
        else:
            detections = detections_to_box2d_training(processed_results)

        if self.count % 100 == 0:
            from openmt.vis.image import imshow_bboxes

            imshow_bboxes(
                inputs[0].image.tensor[0],
                detections[0],  # type: ignore
                frame_id=self.count,
            )
        self.count += 1
        return (
            images,
            features,
            detections,
            detections,
            {**detect_losses},
        )


if __name__ == "__main__":
    d2retinanet_detector_cfg = dict(
        type="D2RetinaNetDetector",
        num_classes=8,
    )
    conf = config.Config(
        model=dict(
            type="DetectorWrapper",
            detection=BaseDetectorConfig(**d2retinanet_detector_cfg),
        ),
        solver=config.Solver(
            images_per_gpu=1,
            lr_policy="WarmupMultiStepLR",
            base_lr=0.0006,
            # steps=[5000, 8000],
            max_iters=10000,
            log_period=10,
            checkpoint_period=100,
            eval_period=500,
            eval_metrics=["detect"],
        ),
        dataloader=Dataloader(
            workers_per_gpu=8,
            ref_sampling_cfg=dict(type="uniform", scope=1, num_ref_imgs=0),
            categories=[
                "pedestrian",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
            ],
            remove_samples_without_labels=True,
        ),
        train=[
            config.Dataset(
                name="bdd100k_sample_train",
                type="BDD100K",
                annotations="openmt/engine/testcases/track/bdd100k-samples/"
                "labels",
                data_root="openmt/engine/testcases/track/bdd100k-samples/"
                "images/",
                # annotations="data/one_sequence/labels",
                # data_root="data/one_sequence/images/",
                # annotations="data/bdd100k/labels/box_track_20/train/",
                # data_root="data/bdd100k/images/track/train/",
                config_path="box_track",
            )
        ],
        test=[
            config.Dataset(
                name="bdd100k_sample_val",
                type="BDD100K",
                annotations="openmt/engine/testcases/track/bdd100k-samples/"
                "labels",
                data_root="openmt/engine/testcases/track/bdd100k-samples/"
                "images/",
                # annotations="data/one_sequence/labels",
                # data_root="data/one_sequence/images/",
                # annotations="data/bdd100k/labels/box_track_20/val/",
                # data_root="data/bdd100k/images/track/val/",
                config_path="box_track",
            )
        ],
    )
    import os
    import shutil

    if os.path.exists("visualization/"):
        shutil.rmtree("visualization/")
    os.mkdir("visualization/")

    conf.launch.weights = "/home/yinjiang/systm/openmt-workspace/DetectorWrapper/2021-07-07_19:34:01/model_0004999.pth"  # self_trained weight
    # conf.launch.weights = "/home/yinjiang/systm/weight/model_final_5bd44e.pkl" # detectron pretrained weight
    # conf.launch.resume = True
    conf.launch.device = "cuda"
    conf.launch.num_gpus = 1

    # conf.launch.resume = True
    train(conf)
