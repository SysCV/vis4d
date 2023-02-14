"""Default data connector configurations for the Mask RCNN model."""
from __future__ import annotations

from vis4d.data.const import CommonKeys
from vis4d.engine.connectors import SourceKeyDescription, data_key, pred_key


def loss_conn() -> dict[str, SourceKeyDescription]:
    """Returns the loss connections for the Mask RCNN model.

    This provides the mapping from the model output to the loss function.

    Returns:
        dict[str, SourceKeyDescription]: The loss connections.
    """
    loss = {}
    # RPN Loss connections
    loss["cls_outs"] = pred_key("rpn.cls")
    loss["reg_outs"] = pred_key("rpn.box")
    loss["target_boxes"] = data_key("boxes2d")
    loss["images_hw"] = data_key("input_hw")

    # RCNN Loss connections
    loss["class_outs"] = pred_key("roi.cls_score")
    loss["regression_outs"] = pred_key("roi.bbox_pred")
    loss["boxes"] = pred_key("sampled_proposals.boxes")
    loss["boxes_mask"] = pred_key("sampled_targets.labels")
    loss["target_boxes"] = pred_key("sampled_targets.boxes")
    loss["target_classes"] = pred_key("sampled_targets.classes")
    loss["pred_sampled_proposals"] = pred_key("sampled_proposals")

    return loss


def train_data_conn() -> dict[str, str]:
    """Returns the training data connections for the Mask RCNN model.

    This provides the mapping from the training data to the model input.

    Returns:
        dict[str, str]: The data connections.
    """
    train = {}
    train[CommonKeys.images] = CommonKeys.images
    train[CommonKeys.input_hw] = CommonKeys.input_hw
    train[CommonKeys.boxes2d] = CommonKeys.boxes2d
    train[CommonKeys.boxes2d_classes] = CommonKeys.boxes2d_classes
    return train


def test_data_conn() -> dict[str, str]:
    """Returns the test data connections for the Mask RCNN model.

    This provides the mapping from the test data to the model input.

    Returns:
        dict[str, str]: The data connections.
    """
    test = {}
    test[CommonKeys.images] = CommonKeys.images
    test[CommonKeys.input_hw] = CommonKeys.input_hw
    test[CommonKeys.boxes2d] = CommonKeys.boxes2d
    test[CommonKeys.boxes2d_classes] = CommonKeys.boxes2d_classes
    test["original_hw"] = "original_hw"
    return test


def eval_bbox_conn() -> dict[str, SourceKeyDescription]:
    """Returns the evaluation data connections for the Mask RCNN model.

    This provides the mapping from the model output to the coco bbox evaluator.

    Returns:
        dict[str, SourceKeyDescription]: The data connections.
    """
    # data connector for evaluator
    coco_eval = {}
    coco_eval["coco_image_id"] = data_key("coco_image_id")
    coco_eval["pred_boxes"] = pred_key("boxes")
    coco_eval["pred_scores"] = pred_key("scores")
    coco_eval["pred_classes"] = pred_key("class_ids")
    return coco_eval


def vis_bbox_conn() -> dict[str, SourceKeyDescription]:
    """Returns the visualization data connections for the Mask RCNN model.

    This provides the mapping from the model output to the bounding box
    visualizer.

    Returns:
        dict[str, SourceKeyDescription]: The data connections.
    """
    # data connector for visualizer
    bbox_vis = {}
    bbox_vis["images"] = data_key("images")
    bbox_vis["boxes"] = pred_key("boxes")
    return bbox_vis
