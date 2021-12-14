"""FPN Decode Head."""
decode_head = dict(
    type="FPNHead",
    in_channels=[64, 64, 64, 64],
    in_index=[0, 1, 2, 3],
    feature_strides=[4, 8, 16, 32],
    channels=32,
    dropout_ratio=0.1,
    num_classes=30,
    norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
    align_corners=False,
    loss_decode=dict(
        type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
