# Model and Loss

Since model and loss are deeply related to each other, we usually write them together.

## Model

Since Vis4D is the modular codebase, our model is an `nn.Module` which consists of various functors, i.e. `op`.

Thus, as the example of Faster R-CNN, we can combine a full model as the following config:

```python
    ######################################################
    ##                        MODEL                     ##
    ######################################################
    anchor_generator = class_config(
        AnchorGenerator,
        scales=[8],
        ratios=[0.5, 1.0, 2.0],
        strides=[4, 8, 16, 32, 64],
    )

    rpn_box_encoder, rpn_box_decoder = get_default_rpn_box_codec_cfg()
    rcnn_box_encoder, rcnn_box_decoder = get_default_rcnn_box_codec_cfg()

    box_matcher = class_config(
        MaxIoUMatcher,
        thresholds=[0.5],
        labels=[0, 1],
        allow_low_quality_matches=False,
    )

    box_sampler = class_config(
        RandomSampler, batch_size=512, positive_fraction=0.25
    )

    roi_head = class_config(RCNNHead, num_classes=num_classes)

    faster_rcnn_head = class_config(
        FasterRCNNHead,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        rpn_box_decoder=rpn_box_decoder,
        box_matcher=box_matcher,
        box_sampler=box_sampler,
        roi_head=roi_head,
    )

    model = class_config(
        FasterRCNN,
        num_classes=num_classes,
        basemodel=basemodel,
        faster_rcnn_head=faster_rcnn_head,
        rcnn_box_decoder=rcnn_box_decoder,
        weights=weights,
    )
```

For more details, please check our [model](https://github.com/SysCV/vis4d/tree/main/vis4d/model/).

## Loss Module

In Vis4D, we use `LossModule` to connect the data, model outputs, and loss functions.

As shown in the following example, loss config is the config of `LossModule` which combines multiple loss functions.
Each function has its `LossConnector` to map the input of loss function from data and model outputs.

```python
    ######################################################
    ##                      LOSS                        ##
    ######################################################
    rpn_loss = class_config(
        RPNLoss,
        anchor_generator=anchor_generator,
        box_encoder=rpn_box_encoder,
    )
    rcnn_loss = class_config(
        RCNNLoss, box_encoder=rcnn_box_encoder, num_classes=num_classes
    )

    loss = class_config(
        LossModule,
        losses=[
            {
                "loss": rpn_loss,
                "connector": class_config(
                    LossConnector, key_mapping=CONN_RPN_LOSS_2D
                ),
            },
            {
                "loss": rcnn_loss,
                "connector": class_config(
                    LossConnector, key_mapping=CONN_ROI_LOSS_2D
                ),
            },
        ],
    )
```

For more details, please check [here](https://github.com/SysCV/vis4d/tree/main/vis4d/engine/loss_module.py) to further set the loss weights and names.
