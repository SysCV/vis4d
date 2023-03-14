"""Default data connector configurations for different task."""
from __future__ import annotations

from ml_collections import ConfigDict

from vis4d.config.util import class_config
from vis4d.data.const import CommonKeys
from vis4d.engine.connectors import (
    DataConnectionInfo,
    SourceKeyDescription,
    StaticDataConnector,
)


def default_detection_connector(
    callbacks: dict[str, dict[str, SourceKeyDescription]] | None = None,
) -> ConfigDict:
    """Default data connector for detection task.

    This creates a config object that can be initialized as a
    StaticDataConnector providing mappings for train, test, loss,
    visualization end evaluation.

    Use the 'evaluators' argument to provide additional, dataset dependant
    evaluators.

    Args:
        callbacks (dict[str, dict[str, SourceKeyDescription]], optional):
            Mapping from callback name to key, value remapping for this
            callback.

    Returns:
        ConfigDict: Config dict that can be instantiated as Data Connector.

    Example:
    >>> # Set up coco evaluator mapping
    >>> evaluator_cfg =
    >>>   coco_eval = {}
    >>> coco_eval["coco_image_id"] = SourceKeyDescription(
    >>>     key="coco_image_id", source="data"
    >>> )
    >>> coco_eval["pred_boxes"] = SourceKeyDescription(
    >>>     key="pred_boxes", source="prediction"
    >>> )
    >>> coco_eval["pred_scores"] = SourceKeyDescription(
    >>>     key="pred_scores", source="prediction"
    >>> )
    >>> # get default detector
    >>> cfg = default_detection_connector(dict(coco = evaluator_cfg))
    """
    train = {}
    train[CommonKeys.images] = CommonKeys.images
    train[CommonKeys.input_hw] = CommonKeys.input_hw
    train[CommonKeys.boxes2d] = CommonKeys.boxes2d
    train[CommonKeys.boxes2d_classes] = CommonKeys.boxes2d_classes

    test = {}
    test[CommonKeys.images] = CommonKeys.images
    test[CommonKeys.input_hw] = CommonKeys.input_hw
    test[CommonKeys.boxes2d] = CommonKeys.boxes2d
    test[CommonKeys.boxes2d_classes] = CommonKeys.boxes2d_classes
    test["original_hw"] = "original_hw"

    loss = {}

    # TODO, remove hardcoded loss connections
    # RPN Loss connections
    loss["cls_outs"] = SourceKeyDescription(key="rpn.cls", source="prediction")
    loss["reg_outs"] = SourceKeyDescription(key="rpn.box", source="prediction")
    loss["target_boxes"] = SourceKeyDescription(key="boxes2d", source="data")
    loss["images_hw"] = SourceKeyDescription(key="input_hw", source="data")

    # RCNN Loss connections
    loss["class_outs"] = SourceKeyDescription(
        key="roi.cls_score", source="prediction"
    )
    loss["regression_outs"] = SourceKeyDescription(
        key="roi.bbox_pred", source="prediction"
    )
    loss["boxes"] = SourceKeyDescription(
        key="sampled_proposals.boxes", source="prediction"
    )
    loss["boxes_mask"] = SourceKeyDescription(
        key="sampled_targets.labels", source="prediction"
    )
    loss["target_boxes"] = SourceKeyDescription(
        key="sampled_targets.boxes", source="prediction"
    )
    loss["target_classes"] = SourceKeyDescription(
        key="sampled_targets.classes", source="prediction"
    )

    loss["pred_sampled_proposals"] = SourceKeyDescription(
        key="sampled_proposals", source="prediction"
    )

    # Visualizer
    info = DataConnectionInfo(train=train, test=test, loss=loss)
    if callbacks is not None:
        info["callbacks"] = callbacks

    return class_config(StaticDataConnector, connections=info)
