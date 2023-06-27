"""MlvlPointGenerator test file.

Modified from mmdetection (https://github.com/open-mmlab/mmdetection).
"""
import pytest

from vis4d.op.box.anchor import MlvlPointGenerator


def test_points_generator():
    """Test MlvlPointGenerator."""
    # test init
    point_generator = MlvlPointGenerator(strides=[4, 8], offset=0)
    assert point_generator.num_base_priors == [1, 1]

    # Square strides
    mlvl_points = MlvlPointGenerator(strides=[4, 10], offset=0)
    mlvl_points_half_stride = MlvlPointGenerator(strides=[4, 10], offset=0.5)
    assert mlvl_points.num_levels == 2

    # assert self.num_levels == len(featmap_sizes)
    with pytest.raises(AssertionError):
        mlvl_points.grid_priors(featmap_sizes=[(2, 2)], device="cpu")
    priors = mlvl_points.grid_priors(
        featmap_sizes=[(2, 2), (4, 8)], device="cpu"
    )
    priors_with_stride = mlvl_points.grid_priors(
        featmap_sizes=[(2, 2), (4, 8)], with_stride=True, device="cpu"
    )
    assert len(priors) == 2

    # assert last dimension is (coord_x, coord_y, stride_w, stride_h).
    assert priors_with_stride[0].size(1) == 4
    assert priors_with_stride[0][0][2] == 4
    assert priors_with_stride[0][0][3] == 4
    assert priors_with_stride[1][0][2] == 10
    assert priors_with_stride[1][0][3] == 10

    stride_4_feat_2_2 = priors[0]
    assert (stride_4_feat_2_2[1] - stride_4_feat_2_2[0]).sum() == 4
    assert stride_4_feat_2_2.size(0) == 4
    assert stride_4_feat_2_2.size(1) == 2

    stride_10_feat_4_8 = priors[1]
    assert (stride_10_feat_4_8[1] - stride_10_feat_4_8[0]).sum() == 10
    assert stride_10_feat_4_8.size(0) == 4 * 8
    assert stride_10_feat_4_8.size(1) == 2

    # assert the offset of 0.5 * stride
    priors_half_offset = mlvl_points_half_stride.grid_priors(
        featmap_sizes=[(2, 2), (4, 8)], device="cpu"
    )

    assert (priors_half_offset[0][0] - priors[0][0]).sum() == 4 * 0.5 * 2
    assert (priors_half_offset[1][0] - priors[1][0]).sum() == 10 * 0.5 * 2
