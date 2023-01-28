"""Default data connector configurations for different task."""
from __future__ import annotations

from ml_collections import ConfigDict

from vis4d.config.util import class_config
from vis4d.data.const import CommonKeys
from vis4d.engine.connectors import DataConnectionInfo, SourceKeyDescription


def default_detection_connector(
    evaluators: dict[str, dict[str, SourceKeyDescription]] | None = None
) -> ConfigDict:
    """Default data connector for detection task.

    This creates a config object that can be initialized as a
    StaticDataConnector providing mappings for train, test, loss,
    visualization end evaluation.

    Use the 'evaluators' argument to provide additional, dataset dependant
    evaluators.


    Args:
        evaluators (dict[str, dict[str, SourceKeyDescription]], optional):
        Mapping from evaluator name to key, value remapping for this evaluator.

    Returns:
        ConfigDict: Config dict that can be instantiated as Data Connector.

    Example:
    >>> # Set up coco evaluator mapping
    >>> evaluator_cfg =
    >>>   coco_eval = dict()
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
    train = dict()
    train[CommonKeys.images] = CommonKeys.images
    train[CommonKeys.input_hw] = CommonKeys.input_hw
    train[CommonKeys.boxes2d] = CommonKeys.boxes2d
    train[CommonKeys.boxes2d_classes] = CommonKeys.boxes2d_classes

    test = dict()
    test[CommonKeys.images] = CommonKeys.images
    test[CommonKeys.input_hw] = CommonKeys.input_hw
    test[CommonKeys.boxes2d] = CommonKeys.boxes2d
    test[CommonKeys.boxes2d_classes] = CommonKeys.boxes2d_classes
    test["original_hw"] = "original_hw"

    loss = dict()
    loss[CommonKeys.input_hw] = SourceKeyDescription(
        key=CommonKeys.input_hw, source="data"
    )
    loss[CommonKeys.boxes2d] = SourceKeyDescription(
        key=CommonKeys.boxes2d, source="data"
    )

    box_vis = dict()
    box_vis[CommonKeys.boxes2d] = SourceKeyDescription(
        key=CommonKeys.boxes2d, source="prediction"
    )
    info = DataConnectionInfo(
        train=train, test=test, loss=loss, vis=dict(boxes=box_vis)
    )
    if evaluators is not None:
        info["evaluators"] = evaluators

    return class_config(
        "vis4d.data.connectors.StaticDataConnector", connections=info
    )
