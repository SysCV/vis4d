"""model settings."""
norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    pretrained="open-mmlab://resnet18_v1c",
    backbone=dict(
        type="ResNetV1c",
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style="pytorch",
        contract_dilation=True,
    ),
    neck=dict(
        type="FPN",
        in_channels=[64, 128, 256, 512],
        out_channels=64,
        num_outs=4,
    ),
    decode_head=dict(
        type="FPNHead",
        in_channels=[64, 64, 64, 64],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=32,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
        ),
    ),
    auxiliary_head=dict(
        type="FPNHead",
        in_channels=[64, 64, 64, 64],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=32,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
        ),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
