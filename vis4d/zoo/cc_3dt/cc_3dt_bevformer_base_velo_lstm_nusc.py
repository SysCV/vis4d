# pylint: disable=duplicate-code
"""CC-3DT with BEV detector on nuScenes."""
from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.typing import DataConfig, ExperimentConfig
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.nuscenes import NuScenes
from vis4d.data.datasets.nuscenes_detection import NuScenesDetection
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.connectors import MultiSensorDataConnector, data_key
from vis4d.model.motion.velo_lstm import VeloLSTM
from vis4d.model.track3d.cc_3dt import CC3DT
from vis4d.op.base import ResNet
from vis4d.op.track3d.cc_3dt import CC3DTrackAssociation
from vis4d.state.track3d.cc_3dt import CC3DTrackGraph
from vis4d.zoo.cc_3dt.cc_3dt_frcnn_r101_fpn_velo_lstm_24e_nusc import (
    get_config as get_velo_lstm_cfg,
)
from vis4d.zoo.cc_3dt.data import CONN_NUSC_BBOX_3D_TEST, get_test_dataloader

CONN_NUSC_BBOX_3D_TEST = {
    "images_list": data_key(K.images, sensors=NuScenes.CAMERAS),
    "images_hw": data_key(K.original_hw, sensors=NuScenes.CAMERAS),
    "intrinsics_list": data_key(K.intrinsics, sensors=NuScenes.CAMERAS),
    "extrinsics_list": data_key(K.extrinsics, sensors=NuScenes.CAMERAS),
    "frame_ids": K.frame_ids,
    "pred_boxes3d": data_key("pred_boxes3d", sensors=["LIDAR_TOP"]),
    "pred_boxes3d_classes": data_key(
        "pred_boxes3d_classes", sensors=["LIDAR_TOP"]
    ),
    "pred_boxes3d_scores": data_key(
        "pred_boxes3d_scores", sensors=["LIDAR_TOP"]
    ),
    "pred_boxes3d_velocities": data_key(
        "pred_boxes3d_velocities", sensors=["LIDAR_TOP"]
    ),
}


def get_config() -> ExperimentConfig:
    """Returns the config dict for CC-3DT on nuScenes.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_velo_lstm_cfg().ref_mode()

    config.experiment_name = "cc_3dt_bevformer_base_velo_lstm_nusc"

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    config.pure_detection = ""

    data = DataConfig()

    data.train_dataloader = None

    test_dataset = class_config(
        NuScenesDetection,
        data_root="data/nuscenes",
        version="v1.0-trainval",
        split="val",
        keys_to_load=[K.images, K.original_images, K.boxes3d],
        data_backend=class_config(HDF5Backend),
        pure_detection=config.pure_detection,
        cache_as_binary=True,
        cached_file_path="data/nuscenes/val.pkl",
    )

    data.test_dataloader = get_test_dataloader(
        test_dataset=test_dataset, samples_per_gpu=1, workers_per_gpu=4
    )

    config.data = data

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    basemodel = class_config(
        ResNet, resnet_name="resnet101", pretrained=True, trainable_layers=3
    )

    track_graph = class_config(
        CC3DTrackGraph,
        track=class_config(
            CC3DTrackAssociation, init_score_thr=0.2, obj_score_thr=0.1
        ),
        motion_model="VeloLSTM",
        lstm_model=class_config(VeloLSTM, weights=config.velo_lstm_ckpt),
        update_3d_score=False,
        add_backdrops=False,
    )

    config.model = class_config(
        CC3DT,
        basemodel=basemodel,
        track_graph=track_graph,
        detection_range=[40, 40, 40, 50, 50, 50, 50, 50, 30, 30],
    )

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.test_data_connector = class_config(
        MultiSensorDataConnector, key_mapping=CONN_NUSC_BBOX_3D_TEST
    )

    return config.value_mode()
