# Connector

In Vis4D, we use the connector design to connect different modules, allowing flexibitly to map the input keys from different sources.

## Data connector

Defines which data to pass to which component.

It extracts the required data from a 'DictData' objects and passes it to the next component with the provided new key.

```python
CONN_BBOX_2D_TRAIN = {
    "images": K.images,
    "input_hw": K.input_hw,
    "boxes2d": K.boxes2d,
    "boxes2d_classes": K.boxes2d_classes,
}

CONN_BBOX_2D_TEST = {
    "images": K.images,
    "input_hw": K.input_hw,
    "original_hw": K.original_hw,
}
```

The key of the key_mapping is the input key of the target function and the value is the key of data dict.

## Loss connector

Defines which data to pass to loss module of the training pipeline.

It extracts the required data from prediction and data and passes it to the next component with the provided new key.

```python
CONN_RPN_LOSS_2D = {
    "cls_outs": pred_key("rpn.cls"),
    "reg_outs": pred_key("rpn.box"),
    "target_boxes": data_key("boxes2d"),
    "images_hw": data_key("input_hw"),
}

CONN_ROI_LOSS_2D = {
    "class_outs": pred_key("roi.cls_score"),
    "regression_outs": pred_key("roi.bbox_pred"),
    "boxes": pred_key("sampled_proposals.boxes"),
    "boxes_mask": pred_key("sampled_targets.labels"),
    "target_boxes": pred_key("sampled_targets.boxes"),
    "target_classes": pred_key("sampled_targets.classes"),
}
```

The key of the key_mapping is the input key of the target function.
As for the value, if `pred_key` means from model outputs and `data_key` means from data dict.

## Callback connector

It extracts the required data from prediction and datas and passes it to the next component with the provided new key.

```python
CONN_COCO_BBOX_EVAL = {
    "coco_image_id": data_key(K.sample_names),
    "pred_boxes": pred_key("boxes"),
    "pred_scores": pred_key("scores"),
    "pred_classes": pred_key("class_ids"),
}
```

It shares the same concept as Loss connector.

For more details, please check [here](https://github.com/SysCV/vis4d/tree/main/vis4d/engine/connectors)