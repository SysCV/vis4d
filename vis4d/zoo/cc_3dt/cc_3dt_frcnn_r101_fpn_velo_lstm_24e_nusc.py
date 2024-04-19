"""CC-3DT inference with Faster-RCNN ResNet-101 detector using VeloLSTM."""

from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig
from vis4d.data.datasets.nuscenes import nuscenes_class_map
from vis4d.model.motion.velo_lstm import VeloLSTM
from vis4d.op.base import ResNet
from vis4d.zoo.cc_3dt.cc_3dt_frcnn_r101_fpn_kf3d_24e_nusc import (
    get_config as get_kf3d_cfg,
)
from vis4d.zoo.cc_3dt.model import get_cc_3dt_cfg


def get_config() -> ExperimentConfig:
    """Returns the config dict for cc-3dt on nuScenes.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_kf3d_cfg().ref_mode()

    config.experiment_name = "cc_3dt_frcnn_r101_fpn_velo_lstm_24e_nusc"

    config.velo_lstm_ckpt = ""

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    basemodel = class_config(
        ResNet, resnet_name="resnet101", pretrained=True, trainable_layers=3
    )

    config.model, _ = get_cc_3dt_cfg(
        num_classes=len(nuscenes_class_map),
        basemodel=basemodel,
        motion_model="VeloLSTM",
        lstm_model=class_config(VeloLSTM, weights=config.velo_lstm_ckpt),
        fps=2,
    )

    return config.value_mode()
