"""CC-3DT with Faster-RCNN ResNet-101 detector generating pure detection."""
from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.default import get_default_callbacks_cfg
from vis4d.config.typing import ExperimentConfig
from vis4d.data.datasets.nuscenes import (
    nuscenes_class_map,
    nuscenes_detection_range_map,
)
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.op.base import ResNet
from vis4d.zoo.cc_3dt.cc_3dt_frcnn_r101_fpn_kf3d_24e_nusc import (
    get_config as get_base_config,
)
from vis4d.zoo.cc_3dt.data import get_nusc_cfg
from vis4d.zoo.cc_3dt.model import get_cc_3dt_cfg


def get_config() -> ExperimentConfig:
    """Get config."""
    config = get_base_config().ref_mode()

    config.experiment_name = "cc_3dt_frcnn_r101_fpn_pure_det_nusc"

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_root = "data/nuscenes"
    version = "v1.0-trainval"
    train_split = "train"
    test_split = "train"

    data_backend = class_config(HDF5Backend)

    config.data = get_nusc_cfg(
        data_root=data_root,
        version=version,
        train_split=train_split,
        test_split=test_split,
        data_backend=data_backend,
        # batch_normalize_images=True,  # Turn on for using old checkpoints
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    basemodel = class_config(
        ResNet, resnet_name="resnet101", pretrained=True, trainable_layers=3
    )

    nuscenes_detection_range = [
        nuscenes_detection_range_map[k] for k in nuscenes_class_map
    ]
    config.model, _ = get_cc_3dt_cfg(
        num_classes=len(nuscenes_class_map),
        basemodel=basemodel,
        detection_range=nuscenes_detection_range,
        fps=2,
        pure_det=True,
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_default_callbacks_cfg(config.output_dir)

    config.callbacks = callbacks

    return config.value_mode()
