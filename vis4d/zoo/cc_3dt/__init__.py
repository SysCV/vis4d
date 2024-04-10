"""CC-3DT Model Zoo."""

from . import (
    cc_3dt_frcnn_r50_fpn_kf3d_12e_nusc,
    cc_3dt_frcnn_r101_fpn_kf3d_24e_nusc,
    cc_3dt_frcnn_r101_fpn_velo_lstm_24e_nusc,
    cc_3dt_nusc_vis,
)

AVAILABLE_MODELS = {
    "cc_3dt_frcnn_r50_fpn_kf3d_12e_nusc": cc_3dt_frcnn_r50_fpn_kf3d_12e_nusc,
    "cc_3dt_frcnn_r101_fpn_kf3d_24e_nusc": cc_3dt_frcnn_r101_fpn_kf3d_24e_nusc,
    "cc_3dt_frcnn_r101_fpn_velo_lstm_24e_nusc": cc_3dt_frcnn_r101_fpn_velo_lstm_24e_nusc,  # pylint: disable=line-too-long
    "cc_3dt_nusc_vis": cc_3dt_nusc_vis,
}
