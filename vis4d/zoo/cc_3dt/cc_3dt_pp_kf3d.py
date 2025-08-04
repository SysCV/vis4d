# pylint: disable=duplicate-code
"""CC-3DT++ on nuScenes."""
from __future__ import annotations

from vis4d.config import class_config
from vis4d.zoo.base import get_default_callbacks_cfg
from vis4d.config.typing import DataConfig, ExperimentConfig
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.nuscenes import NuScenes
from vis4d.data.datasets.nuscenes_detection import NuScenesDetection
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.connectors import MultiSensorDataConnector, data_key
from vis4d.model.track3d.cc_3dt import CC3DT
from vis4d.op.base import ResNet
from vis4d.op.track3d.cc_3dt import CC3DTrackAssociation
from vis4d.state.track3d.cc_3dt import CC3DTrackGraph
from vis4d.engine.callbacks import EvaluatorCallback
from vis4d.engine.connectors import (
    CallbackConnector,
    MultiSensorDataConnector,
    data_key,
)
from vis4d.eval.nuscenes import (
    NuScenesDet3DEvaluator,
    NuScenesTrack3DEvaluator,
)

from vis4d.zoo.cc_3dt.cc_3dt_frcnn_r101_fpn_kf3d_24e_nusc import (
    get_config as get_kf3d_cfg,
)
from vis4d.zoo.cc_3dt.data import (
    CONN_NUSC_DET3D_EVAL,
    CONN_NUSC_TRACK3D_EVAL,
    get_test_dataloader,
)

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
    config = get_kf3d_cfg().ref_mode()

    config.experiment_name = "cc_3dt_pp_kf3d_nusc"

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    config.pure_detection = ""

    data_root = "data/nuscenes"
    version = "v1.0-trainval"
    test_split = "val"

    data = DataConfig()

    data.train_dataloader = None

    test_dataset = class_config(
        NuScenesDetection,
        data_root=data_root,
        version=version,
        split=test_split,
        keys_to_load=[K.images, K.original_images],
        data_backend=class_config(HDF5Backend),
        pure_detection=config.pure_detection,
        cache_as_binary=True,
        cached_file_path=f"{data_root}/val.pkl",
    )

    data.test_dataloader = get_test_dataloader(
        test_dataset=test_dataset, samples_per_gpu=1, workers_per_gpu=1
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
            CC3DTrackAssociation,
            init_score_thr=0.2,
            obj_score_thr=0.1,
            match_score_thr=0.3,
            nms_class_iou_thr=0.3,
            bbox_affinity_weight=0.75,
            with_velocities=True,
        ),
        update_3d_score=False,
        use_velocities=True,
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

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_default_callbacks_cfg(config.output_dir)

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                NuScenesDet3DEvaluator,
                data_root=data_root,
                version=version,
                split=test_split,
            ),
            save_predictions=True,
            output_dir=config.output_dir,
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_NUSC_DET3D_EVAL
            ),
        )
    )

    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                NuScenesTrack3DEvaluator, metadata=("use_camera", "use_radar")
            ),
            save_predictions=True,
            output_dir=config.output_dir,
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_NUSC_TRACK3D_EVAL
            ),
        )
    )

    config.callbacks = callbacks

    return config.value_mode()
