"""Common models."""
from vis4d.model.detect.mmdet import MMOneStageDetector, MMTwoStageDetector
from vis4d.model.segment import MMEncDecSegmentor
from vis4d.struct import CategoryMap, DictStrAny


def build_faster_rcnn(  # pylint: disable=dangerous-default-value
    category_mapping: CategoryMap,
    backbone: str = "r50_fpn",
    model_kwargs: DictStrAny = {},
) -> MMTwoStageDetector:
    """Build a default Faster-RCNN detector."""
    faster_rcnn = MMTwoStageDetector(
        category_mapping=category_mapping,
        model_base=f"mmdet://faster_rcnn/faster_rcnn_{backbone}_1x_coco.py",
        model_kwargs={
            "rpn_head.loss_bbox.type": "SmoothL1Loss",
            "rpn_head.loss_bbox.beta": 0.111,
            "roi_head.bbox_head.loss_bbox.type": "SmoothL1Loss",
        },
        pixel_mean=(123.675, 116.28, 103.53),
        pixel_std=(58.395, 57.12, 57.375),
        backbone_output_names=["p2", "p3", "p4", "p5", "p6"],
        **model_kwargs,
    )
    return faster_rcnn


def build_mask_rcnn(
    category_mapping: CategoryMap, backbone: str = "r50_fpn"
) -> MMTwoStageDetector:
    """Build a default Mask-RCNN network."""
    faster_rcnn = MMTwoStageDetector(
        category_mapping=category_mapping,
        model_base=f"mmdet://mask_rcnn/mask_rcnn_{backbone}_1x_coco.py",
        pixel_mean=(123.675, 116.28, 103.53),
        pixel_std=(58.395, 57.12, 57.375),
        backbone_output_names=["p2", "p3", "p4", "p5", "p6"],
    )
    return faster_rcnn


def build_retinanet(  # pylint: disable=dangerous-default-value
    category_mapping: CategoryMap,
    backbone: str = "r50_fpn",
    model_kwargs: DictStrAny = {},
) -> MMOneStageDetector:
    """Build a default RetinaNet detector."""
    retinanet = MMOneStageDetector(
        category_mapping=category_mapping,
        model_base=f"mmdet://retinanet/retinanet_{backbone}_1x_coco.py",
        pixel_mean=(123.675, 116.28, 103.53),
        pixel_std=(58.395, 57.12, 57.375),
        backbone_output_names=["p2", "p3", "p4", "p5", "p6"],
        **model_kwargs,
    )
    return retinanet


def build_yolox(  # pylint: disable=dangerous-default-value
    category_mapping: CategoryMap,
    version: str = "yolox_x",
    model_kwargs: DictStrAny = {},
) -> MMOneStageDetector:
    """Build a default YOLOX detector."""
    yolox = MMOneStageDetector(
        category_mapping=category_mapping,
        model_base=f"mmdet://yolox/{version}_8x8_300e_coco.py",
        model_kwargs={
            "test_cfg": {
                "score_thr": 0.001,
                "nms": {"type": "nms", "iou_threshold": 0.7},
            },
        },
        image_channel_mode="BGR",
        pixel_mean=(0.0, 0.0, 0.0),
        pixel_std=(1.0, 1.0, 1.0),
        weights="mmdet://yolox/yolox_x_8x8_300e_coco/"
        "yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth",
        **model_kwargs,
    )
    return yolox


def build_deeplabv3plus(  # pylint: disable=dangerous-default-value
    category_mapping: CategoryMap,
    backbone: str = "r50-d8",
    model_kwargs: DictStrAny = {},
) -> MMEncDecSegmentor:
    """Build a default DeepLabv3+ segmentor."""
    deeplabv3plus = MMEncDecSegmentor(
        category_mapping=category_mapping,
        model_base=f"mmseg://deeplabv3plus/deeplabv3plus_{backbone}_512x1024_80k_cityscapes.py",  # pylint: disable=line-too-long
        pixel_mean=(123.675, 116.28, 103.53),
        pixel_std=(58.395, 57.12, 57.375),
        **model_kwargs,
    )
    return deeplabv3plus


def build_semantic_fpn(  # pylint: disable=dangerous-default-value
    category_mapping: CategoryMap,
    backbone: str = "r50",
    model_kwargs: DictStrAny = {},
) -> MMEncDecSegmentor:
    """Build a default SemanticFPN segmentor."""
    semantic_fpn = MMEncDecSegmentor(
        category_mapping=category_mapping,
        model_base=f"mmseg://sem_fpn/fpn_{backbone}_512x1024_80k_cityscapes.py",  # pylint: disable=line-too-long
        pixel_mean=(123.675, 116.28, 103.53),
        pixel_std=(58.395, 57.12, 57.375),
        **model_kwargs,
    )
    return semantic_fpn
