"""Linear Cls Head."""
head = dict(
    type="LinearClsHead",
    num_classes=7,
    in_channels=2048,
    loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    topk=(1, 5),
)