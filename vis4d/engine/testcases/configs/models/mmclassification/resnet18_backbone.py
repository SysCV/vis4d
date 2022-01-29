"""ResNet-18 Backbone."""
backbone = dict(
    type="ResNet", depth=18, num_stages=4, out_indices=(3,), style="pytorch"
)
