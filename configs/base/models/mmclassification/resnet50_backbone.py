"""ResNet-50 Backbone."""
backbone = dict(
    type="ResNet", depth=50, num_stages=4, out_indices=(3,), style="pytorch"
)
