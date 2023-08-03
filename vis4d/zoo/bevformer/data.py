"""CC-3DT NuScenes data config."""
from __future__ import annotations

from ml_collections import ConfigDict

from vis4d.config import class_config
from vis4d.config.common.datasets.nuscenes import (
    get_nusc_mini_train_cfg,
    get_nusc_mini_val_cfg,
    get_nusc_train_cfg,
    get_nusc_val_cfg,
)
from vis4d.config.typing import DataConfig
from vis4d.config.util import (
    get_inference_dataloaders_cfg,
    get_train_dataloader_cfg,
)
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets.nuscenes import NuScenes
from vis4d.data.loader import multi_sensor_collate
from vis4d.data.reference import MultiViewDataset, UniformViewSampler
from vis4d.data.transforms import RandomApply, compose
from vis4d.data.transforms.flip import (
    FlipBoxes2D,
    FlipBoxes3D,
    FlipImages,
    FlipIntrinsics,
)
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.post_process import PostProcessBoxes2D
from vis4d.data.transforms.resize import (
    GenResizeParameters,
    ResizeBoxes2D,
    ResizeImages,
    ResizeIntrinsics,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.engine.connectors import data_key, pred_key


nuscenes_class_map = {
    "car": 0,
    "truck": 1,
    "construction_vehicle": 2,
    "bus": 3,
    "trailer": 4,
    "barrier": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "pedestrian": 8,
    "traffic_cone": 9,
}

NUSC_SENSORS = [
    "LIDAR_TOP",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

NUSC_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

CONN_NUSC_BBOX_3D_TEST = {
    "images": data_key(K.images, sensors=NUSC_CAMERAS),
    "images_hw": data_key(K.input_hw, sensors=NUSC_CAMERAS),
    "can_bus": "can_bus",
    "scene_names": K.sequence_names,
    "cam_intrinsics": data_key(K.intrinsics, sensors=NUSC_CAMERAS),
    "cam_extrinsics": data_key(K.extrinsics, sensors=NUSC_CAMERAS),
    "lidar_extrinsics": data_key(K.extrinsics, sensors=["LIDAR_TOP"]),
}

CONN_NUSC_BBOX_3D_VIS = {
    "images": data_key(K.original_images, sensors=NUSC_CAMERAS),
    "image_names": data_key(K.sample_names, sensors=NUSC_CAMERAS),
    "boxes3d": pred_key("boxes_3d"),
    "intrinsics": data_key(K.intrinsics, sensors=NUSC_CAMERAS),
    "extrinsics": data_key(K.extrinsics, sensors=NUSC_CAMERAS),
    "scores": pred_key("scores_3d"),
    "class_ids": pred_key("class_ids"),
    "sequence_names": data_key(K.sequence_names),
}

CONN_NUSC_DET3D_EVAL = {
    "tokens": data_key("token"),
    "boxes_3d": pred_key("boxes_3d"),
    "velocities": pred_key("velocities"),
    "class_ids": pred_key("class_ids"),
    "scores_3d": pred_key("scores_3d"),
}


def get_train_dataloader(
    train_dataset: ConfigDict, samples_per_gpu: int, workers_per_gpu: int
) -> ConfigDict:
    """Get the default train dataloader for nuScenes tracking."""
    train_dataset_cfg = class_config(
        MultiViewDataset,
        dataset=train_dataset,
        sampler=class_config(UniformViewSampler, scope=2, num_ref_samples=1),
    )

    preprocess_transforms = [
        class_config(GenResizeParameters, shape=(900, 1600), keep_ratio=True),
        class_config(ResizeImages),
        class_config(ResizeBoxes2D),
    ]

    preprocess_transforms.append(
        class_config(
            RandomApply,
            transforms=[
                class_config(FlipImages),
                class_config(FlipIntrinsics),
                class_config(FlipBoxes2D),
                class_config(FlipBoxes3D),
            ],
            probability=0.5,
        )
    )

    preprocess_transforms.append(class_config(PostProcessBoxes2D))

    train_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    train_batchprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(PadImages),
            class_config(NormalizeImages),
            class_config(ToTensor),
        ],
    )

    return get_train_dataloader_cfg(
        preprocess_cfg=train_preprocess_cfg,
        dataset_cfg=train_dataset_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        batchprocess_cfg=train_batchprocess_cfg,
    )


def get_test_dataloader(
    test_dataset: ConfigDict, samples_per_gpu: int, workers_per_gpu: int
) -> ConfigDict:
    """Get the default test dataloader for nuScenes tracking."""
    test_transforms = [
        class_config(
            GenResizeParameters,
            shape=(900, 1600),
            keep_ratio=True,
            sensors=NUSC_CAMERAS,
        ),
        class_config(ResizeImages, sensors=NUSC_CAMERAS),
        class_config(ResizeIntrinsics, sensors=NUSC_CAMERAS),
        class_config(
            NormalizeImages,
            mean=[103.530, 116.280, 123.675],
            std=[1.0, 1.0, 1.0],
            sensors=NUSC_CAMERAS,
        ),
    ]

    test_preprocess_cfg = class_config(compose, transforms=test_transforms)

    test_batch_transforms = [
        class_config(PadImages, sensors=NUSC_CAMERAS),
        class_config(ToTensor, sensors=NUSC_SENSORS),
    ]

    test_batchprocess_cfg = class_config(
        compose, transforms=test_batch_transforms
    )

    test_dataset_cfg = class_config(
        DataPipe, datasets=test_dataset, preprocess_fn=test_preprocess_cfg
    )

    return get_inference_dataloaders_cfg(
        datasets_cfg=test_dataset_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        video_based_inference=True,
        batchprocess_cfg=test_batchprocess_cfg,
        collate_fn=multi_sensor_collate,
        sensors=NUSC_SENSORS,
    )


def get_nusc_cfg(
    data_root: str = "data/nuscenes",
    version: str = "v1.0-trainval",
    train_split: str = "train",
    test_split: str = "val",
    data_backend: None | ConfigDict = None,
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
) -> DataConfig:
    """Get the default config for nuScenes tracking."""
    data = DataConfig()

    if version == "v1.0-mini":
        assert train_split == "mini_train"
        assert test_split == "mini_val"
        train_dataset = get_nusc_mini_train_cfg(
            data_root=data_root,
            data_backend=data_backend,
            cache_as_binary=False,
        )
        test_dataset = get_nusc_mini_val_cfg(
            data_root=data_root,
            image_channel_mode="BGR",
            data_backend=data_backend,
            cached_file_path="data/nuscenes/bevformer_mini_val.pkl",
        )
    elif version == "v1.0-trainval":
        assert train_split == "train"
        train_dataset = get_nusc_train_cfg(
            data_root=data_root, data_backend=data_backend
        )

        if test_split == "val":
            test_dataset = get_nusc_val_cfg(
                data_root=data_root,
                image_channel_mode="BGR",
                data_backend=data_backend,
                cached_file_path="data/nuscenes/bevformer_val.pkl",
            )
        elif test_split == "train":
            test_dataset = get_nusc_train_cfg(
                data_root=data_root,
                skip_empty_samples=False,
                keys_to_load=[K.images, K.original_images, K.boxes3d],
                data_backend=data_backend,
            )
    else:
        # TODO: Add support for v1.0-test
        raise ValueError(f"Unknown version {version}")

    data.train_dataloader = get_train_dataloader(
        train_dataset=train_dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )

    data.test_dataloader = get_test_dataloader(
        test_dataset, samples_per_gpu=1, workers_per_gpu=1
    )

    return data
