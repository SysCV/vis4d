"""model settings."""
norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    # pretrained="open-mmlab://resnet18_v1c",
    backbone=dict(
        type="ResNetV1c",
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style="pytorch",
        contract_dilation=True,
    ),
    decode_head=dict(
        type="DepthwiseSeparableASPPHead",
        in_channels=512,
        in_index=3,
        channels=128,
        dilations=(1, 12, 24, 36),
        c1_in_channels=64,
        c1_channels=24,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
        ),
    ),
    auxiliary_head=[
        dict(
            type="FCNHead",
            in_channels=256,
            in_index=2,
            channels=64,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        ),
        dict(
            type="FCNHead",
            in_channels=256,
            in_index=2,
            channels=64,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        ),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
