"""NuScenes multi-sensor video dataset."""
from __future__ import annotations

from tqdm import tqdm

import os
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch

from torch import Tensor

from vis4d.common.imports import NUSCENES_AVAILABLE
from vis4d.common.typing import DictStrAny, NDArrayF32, NDArrayI64, NDArrayBool
from vis4d.data.const import AxisMode
from vis4d.data.const import CommonKeys as K
from vis4d.data.io import DataBackend, FileBackend

from vis4d.data.typing import DictData
from vis4d.op.geometry.transform import inverse_rigid_transform
from vis4d.op.geometry.projection import generate_depth_map

from scipy.spatial.transform import Rotation as R

from .base import VideoDataset
from .util import im_decode, CacheMappingMixin

if NUSCENES_AVAILABLE:
    from nuscenes import NuScenes as NuScenesDevkit
    from nuscenes.eval.detection.utils import category_to_detection_name
    from nuscenes.utils.data_classes import Quaternion
    from nuscenes.utils.geometry_utils import (
        transform_matrix,
        view_points,
        box_in_image,
    )
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes.scripts.export_2d_annotations_as_json import (
        post_process_coords,
    )

nuscenes_class_map = {
    "bicycle": 0,
    "motorcycle": 1,
    "pedestrian": 2,
    "bus": 3,
    "car": 4,
    "trailer": 5,
    "truck": 6,
    "construction_vehicle": 7,
    "traffic_cone": 8,
    "barrier": 9,
}

nuscenes_attribute_map = {
    "cycle.with_rider": 0,
    "cycle.without_rider": 1,
    "pedestrian.moving": 2,
    "pedestrian.standing": 3,
    "pedestrian.sitting_lying_down": 4,
    "vehicle.moving": 5,
    "vehicle.parked": 6,
    "vehicle.stopped": 7,
    "": 8,
}

nuscenes_detection_range = [40, 40, 40, 50, 50, 50, 50, 50, 30, 30]

nuscenes_detection_range_map = {
    "bicycle": 40,
    "motorcycle": 40,
    "pedestrian": 40,
    "bus": 50,
    "car": 50,
    "trailer": 50,
    "truck": 50,
    "construction_vehicle": 50,
    "traffic_cone": 30,
    "barrier": 30,
}


def _get_extrinsics(
    ego_pose: DictStrAny, car_from_sensor: DictStrAny
) -> NDArrayF32:
    """Get NuScenes sensor to global extrinsics."""
    global_from_car = transform_matrix(
        ego_pose["translation"],
        Quaternion(ego_pose["rotation"]),
        inverse=False,
    )
    car_from_sensor_ = transform_matrix(
        car_from_sensor["translation"],
        Quaternion(car_from_sensor["rotation"]),
        inverse=False,
    )
    extrinsics = np.dot(global_from_car, car_from_sensor_).astype(np.float32)
    return extrinsics


class NuScenes(CacheMappingMixin, VideoDataset):
    """NuScenes multi-sensor video dataset.

    This dataset loads both LiDAR and camera inputs from the NuScenes dataset
    into the Vis4D expected format for multi-sensor, video datasets.
    """

    DESCRIPTION = "NuScenes multi-sensor driving video dataset."
    HOMEPAGE = "https://www.nuscenes.org/"
    PAPER = "https://arxiv.org/abs/1903.11027"
    LICENSE = "https://www.nuscenes.org/license"

    KEYS = [
        K.images,
        K.original_hw,
        K.input_hw,
        K.intrinsics,
        K.extrinsics,
        K.timestamp,
        K.axis_mode,
        K.boxes2d,
        K.boxes2d_classes,
        K.boxes2d_track_ids,
        K.boxes3d,
        K.boxes3d_classes,
        K.boxes3d_track_ids,
    ]

    _SENSORS = [
        "LIDAR_TOP",
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    _CAMERAS = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        data_root: str,
        keys_to_load: Sequence[str] = (
            K.images,
            K.boxes2d,
            K.boxes3d,
        ),
        sensors: Sequence[str] = (
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ),
        version: str = "v1.0-trainval",
        split: str = "train",
        include_non_key: bool = False,
        data_backend: DataBackend | None = None,
        skip_empty_samples: bool = False,
        point_based_filter: bool = False,
        distance_based_filter: bool = False,
        cache_as_binary: bool = False,
        cached_file_path: str | None = None,
    ) -> None:
        """Creates an instance of the class.

        Args:
            data_root (str): Root directory of nuscenes data in original
                format.
            keys_to_load (tuple[str, ...]): Keys to load from the dataset.
                Defaults to (K.images, K.boxes2d, K.boxes3d).
            sensors (Sequence[str, ...]): Which sensor to load. Defaults
                to ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT").
            version (str, optional): Version of the data to load. Defaults to
                "v1.0-trainval".
            split (str, optional): Split of the data to load. Defaults to
                "train".
            include_non_key (bool, optional): Whether to include non-key
                samples. Note that Non-key samples do not have annotations.
                Defaults to False.
            data_backend (DataBackend | None, optional): Data backend to use
                for loading data. Defaults to None.
            skip_empty_samples (bool, optional): Whether to skip samples
                without annotations. Defaults to False.
            cache_as_binary (bool, optional): Whether to cache the loaded
                data as binary. Defaults to True.
            point_based_filter (bool, optional): Whether to filter out
                samples based on the number of points in the point cloud.
                Defaults to False.
            distance_based_filter (bool, optional): Whether to filter out
                samples based on the distance of the object from the ego
                vehicle. Defaults to False.
        """
        super().__init__()
        self.data_root = data_root
        self.keys_to_load = keys_to_load
        self.sensors = sensors
        self._check_version_and_split(version, split)
        self.include_non_key = include_non_key  # TODO: Add non-key frames
        self.data_backend = (
            FileBackend() if data_backend is None else data_backend
        )
        # TODO: implenment skip_empty_samples for training
        self.skip_empty_samples = skip_empty_samples

        self.point_based_filter = point_based_filter
        self.distance_based_filter = distance_based_filter

        # Load annotations
        self.samples = self._load_mapping(
            self._generate_data_mapping,
            cache_as_binary=cache_as_binary,
            cached_file_path=cached_file_path,
        )

        # Generate video to indices mapping
        self.video_to_indices = self._generate_video_to_indices()

    def _check_version_and_split(self, version: str, split: str) -> None:
        """Check that the version and split are valid."""
        assert version in {
            "v1.0-trainval",
            "v1.0-test",
            "v1.0-mini",
        }, f"Invalid version {version} for NuScenes!"
        self.version = version

        if "mini" in version:
            valid_splits = {"mini_train", "mini_val"}
        elif "test" in version:
            valid_splits = {"test"}
        else:
            valid_splits = {"train", "val"}

        assert (
            split in valid_splits
        ), f"Invalid split {split} for NuScenes {version}!"
        self.split = split

    def __repr__(self) -> str:
        """Concise representation of the dataset."""
        return f"NuScenesDataset {self.version} {self.split}"

    def _generate_video_to_indices(self) -> dict[str, list[int]]:
        """Group dataset sample indices by their associated video ID.

        The sample index is an integer while video IDs are string.

        Returns:
            dict[str, list[int]]: Mapping video to index.
        """
        video_mapping = defaultdict(list)
        for i, sample in enumerate(self.samples):  # type: ignore
            video_mapping[sample["scene_name"]].append(i)
        return video_mapping

    def _generate_data_mapping(self) -> list[DictStrAny]:
        """Generate data mapping.

        Returns:
            List[DictStrAny]: List of items required to load for a single
                dataset sample.
        """
        data = NuScenesDevkit(
            version=self.version, dataroot=self.data_root, verbose=False
        )

        frames = []
        instance_tokens: list[str] = []

        scene_names_per_split = create_splits_scenes()

        scenes = [
            scene
            for scene in data.scene
            if scene["name"] in scene_names_per_split[self.split]
        ]

        for scene in tqdm(scenes):
            scene_name = scene["name"]
            frame_ids = 0
            sample_token = scene["first_sample_token"]
            while sample_token:
                frame = {}
                sample = data.get("sample", sample_token)

                frame["scene_name"] = scene_name
                frame["token"] = sample["token"]
                frame["frame_ids"] = frame_ids

                # TODO: Check the lidar data
                lidar_token = sample["data"]["LIDAR_TOP"]

                frame["LIDAR_TOP"] = self._load_lidar_data(data, lidar_token)
                frame["LIDAR_TOP"]["annotations"] = self._load_annotations(
                    data,
                    frame["LIDAR_TOP"]["extrinsics"],
                    sample["anns"],
                    instance_tokens,
                )

                # Get the sample data for each camera
                for cam in self._CAMERAS:
                    cam_token = sample["data"][cam]

                    frame[cam] = self._load_cam_data(data, cam_token)
                    frame[cam]["annotations"] = self._load_annotations(
                        data,
                        frame[cam]["extrinsics"],
                        sample["anns"],
                        instance_tokens,
                        axis_mode=AxisMode.OPENCV,
                        export_2d_annotations=True,
                        intrinsics=frame[cam]["intrinsics"],
                        image_hw=frame[cam]["image_hw"],
                    )

                # TODO add RADAR, Map data

                frames.append(frame)

                sample_token = sample["next"]
                frame_ids += 1

        return frames

    def _load_lidar_data(
        self, data: NuScenesDevkit, lidar_token: str
    ) -> DictStrAny:
        """Load LiDAR data.

        Args:
            data (NuScenesDevkit): NuScenes toolkit.
            lidar_token (str): LiDAR token.

        Returns:
            DictStrAny: LiDAR data.
        """
        lidar_data = data.get("sample_data", lidar_token)

        sample_name = (
            lidar_data["filename"]
            .split("/")[-1]
            .replace(f".{lidar_data['fileformat']}", "")
        )

        lidar_path = os.path.join(self.data_root, lidar_data["filename"])

        calibration_lidar = data.get(
            "calibrated_sensor", lidar_data["calibrated_sensor_token"]
        )

        ego_pose = data.get("ego_pose", lidar_data["ego_pose_token"])

        extrinsics = _get_extrinsics(ego_pose, calibration_lidar)

        return {
            "sample_name": sample_name,
            "lidar_path": lidar_path,
            "extrinsics": extrinsics,
            "timestamp": lidar_data["timestamp"],
        }

    def _load_cam_data(
        self, data: NuScenesDevkit, cam_token: str
    ) -> DictStrAny:
        """Load camera data.

        Args:
            cam_data (DictStrAny): NuScenes format camera data.
            ego_pose_cam (dict): Ego vehicle pose in NuScenes format.

        Returns:
            tuple[NDArrayF32, NDArrayF32, NDArrayF32, str]: Image, intrinscs,
                extrinsics, timestamp.
        """
        cam_data = data.get("sample_data", cam_token)

        sample_name = (
            cam_data["filename"]
            .split("/")[-1]
            .replace(f".{cam_data['fileformat']}", "")
        )

        image_path = os.path.join(self.data_root, cam_data["filename"])

        calibration_cam = data.get(
            "calibrated_sensor", cam_data["calibrated_sensor_token"]
        )

        intrinsics = np.array(
            calibration_cam["camera_intrinsic"], dtype=np.float32
        )

        ego_pose = data.get("ego_pose", cam_data["ego_pose_token"])
        extrinsics = _get_extrinsics(ego_pose, calibration_cam)

        return {
            "sample_name": sample_name,
            "image_path": image_path,
            "image_hw": (cam_data["height"], cam_data["width"]),
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "timestamp": cam_data["timestamp"],
        }

    # TODO: add unit tests for all coordinate transforms
    def _load_annotations(
        self,
        data: NuScenesDevkit,
        extrinsics: NDArrayF32,
        ann_tokens: list[str],
        instance_tokens: list[str],
        axis_mode: AxisMode = AxisMode.ROS,
        export_2d_annotations: bool = False,
        intrinsics: NDArrayF32 | None = None,
        image_hw: tuple[int, int] | None = None,
    ) -> DictStrAny:
        """Load annonations."""
        boxes3d = np.empty((1, 10), dtype=np.float32)[1:]
        boxes3d_classes = np.empty((1,), dtype=np.int64)[1:]
        boxes3d_attributes = np.empty((1,), dtype=np.int64)[1:]
        boxes3d_track_ids = np.empty((1,), dtype=np.int64)[1:]
        boxes3d_velocities = np.empty((1, 3), dtype=np.float32)[1:]
        boxes3d_num_lidar_pts = np.empty((1,), dtype=np.int64)[1:]
        boxes3d_num_radar_pts = np.empty((1,), dtype=np.int64)[1:]

        if export_2d_annotations:
            assert (
                axis_mode == AxisMode.OPENCV
            ), "2D annotations are only supported in camera coordinates."
            assert intrinsics is not None, "Intrinsics must be provided."
            assert (
                image_hw is not None
            ), "Image height and width must be provided."
            boxes2d = np.empty((1, 4), dtype=np.float32)[1:]

        sensor_from_global = inverse_rigid_transform(
            torch.from_numpy(extrinsics)
        )
        translation = sensor_from_global[:3, 3].numpy()
        rotation = Quaternion(
            matrix=sensor_from_global[:3, :3].numpy(), atol=1e-5
        )

        for ann_token in ann_tokens:
            ann_info = data.get("sample_annotation", ann_token)
            box3d_class = category_to_detection_name(ann_info["category_name"])

            if box3d_class is None:
                continue

            # 3D box in global coordinates
            box3d = data.get_box(ann_info["token"])

            # Get 3D box velocity
            box3d.velocity = data.box_velocity(ann_info["token"])

            # Move 3D box to sensor coordinates
            box3d.rotate(rotation)
            box3d.translate(translation)

            if export_2d_annotations and not box_in_image(
                box3d, intrinsics, (image_hw[1], image_hw[0])
            ):
                continue

            # Number of points in the 3D box
            boxes3d_num_lidar_pts = np.concatenate(
                [
                    boxes3d_num_lidar_pts,
                    np.array([ann_info["num_lidar_pts"]], dtype=np.int64),
                ]
            )
            boxes3d_num_radar_pts = np.concatenate(
                [
                    boxes3d_num_radar_pts,
                    np.array([ann_info["num_radar_pts"]], dtype=np.int64),
                ]
            )

            # Get 2D box
            if export_2d_annotations:
                corner_coords = (
                    view_points(box3d.corners(), intrinsics, True)
                    .T[:, :2]
                    .tolist()
                )

                boxes2d = np.concatenate(
                    [
                        boxes2d,
                        np.array(
                            [post_process_coords(corner_coords)],
                            dtype=np.float32,
                        ),
                    ]
                )

            # Get 3D box orientation
            v = np.dot(box3d.orientation.rotation_matrix, np.array([1, 0, 0]))
            if axis_mode == AxisMode.ROS:
                yaw = box3d.orientation.yaw_pitch_roll[0]
                x, y, z, w = R.from_euler("xyz", [0, 0, yaw]).as_quat()
                orientation = Quaternion([w, x, y, z])
            else:
                yaw = -box3d.orientation.yaw_pitch_roll[0]
                x, y, z, w = R.from_euler("xyz", [0, yaw, 0]).as_quat()
                orientation = Quaternion([w, x, y, z])

            boxes3d = np.concatenate(
                [
                    boxes3d,
                    np.array(
                        [[*box3d.center, *box3d.wlh, *orientation.elements]],
                        dtype=np.float32,
                    ),
                ]
            )

            # Get 3D box class id
            boxes3d_classes = np.concatenate(
                [
                    boxes3d_classes,
                    np.array(
                        [nuscenes_class_map[box3d_class]], dtype=np.int64
                    ),
                ]
            )

            # Get 3D box attribute id
            if len(ann_info["attribute_tokens"]) == 0:
                box3d_attr = ""
            else:
                box3d_attr = data.get(
                    "attribute", ann_info["attribute_tokens"][0]
                )["name"]
            boxes3d_attributes = np.concatenate(
                [
                    boxes3d_attributes,
                    np.array(
                        [nuscenes_attribute_map[box3d_attr]], dtype=np.int64
                    ),
                ]
            )

            # Get 3D box track id
            instance_token = data.get("sample_annotation", box3d.token)[
                "instance_token"
            ]
            if not instance_token in instance_tokens:
                instance_tokens.append(instance_token)
            track_id = instance_tokens.index(instance_token)

            boxes3d_track_ids = np.concatenate(
                [boxes3d_track_ids, np.array([track_id], dtype=np.int64)]
            )

            # 3D bounding box velocity
            velocity = box3d.velocity.astype(np.float32)
            if np.any(np.isnan(velocity)):
                velocity = np.zeros(3, dtype=np.float32)

            boxes3d_velocities = np.concatenate(
                [boxes3d_velocities, velocity[None]]
            )

        annotations = {
            "boxes3d": boxes3d,
            "boxes3d_classes": boxes3d_classes,
            "boxes3d_attributes": boxes3d_attributes,
            "boxes3d_track_ids": boxes3d_track_ids,
            "boxes3d_velocities": boxes3d_velocities,
            "boxes3d_num_lidar_pts": boxes3d_num_lidar_pts,
            "boxes3d_num_radar_pts": boxes3d_num_radar_pts,
        }

        if export_2d_annotations:
            annotations["boxes2d"] = boxes2d

        return annotations

    def _load_depth_map(
        self,
        points_lidar: NDArrayF32,
        lidar2global: NDArrayF32,
        cam2global: NDArrayF32,
        intrinsic: NDArrayF32,
        image_hw: tuple[int, int],
    ) -> Tensor:
        """Load depth map.

        Args:
            points_lidar (Tensor): LiDAR points.
            lidar2global (Tensor): LiDAR to global extrinsics.
            cam2global (Tensor): Camera to global extrinsics.
            image_hw (tuple[int, int]): Image height and width.

        Returns:
            depth_map (Tensor): Depth map.
        """
        cam2global = torch.from_numpy(cam2global)
        lidar2global = torch.from_numpy(lidar2global)
        intrinsic = torch.from_numpy(intrinsic)
        points_lidar = torch.from_numpy(np.copy(points_lidar))

        lidar2cam = torch.matmul(torch.inverse(cam2global), lidar2global)
        cam2img = torch.eye(4, 4)
        cam2img[:3, :3] = intrinsic
        points_cam = points_lidar[:, :3] @ (lidar2cam[:3, :3].T) + lidar2cam[
            :3, 3
        ].unsqueeze(0)

        depth_map = generate_depth_map(
            points_cam,
            intrinsic,
            image_hw,
        )
        depth_map = depth_map.numpy()
        return depth_map

    def _filter_boxes(
        self, annotations: DictStrAny
    ) -> tuple[
        NDArrayBool, NDArrayF32, NDArrayI64, NDArrayI64, NDArrayI64, NDArrayF32
    ]:
        """Load boxes."""
        valid_mask = np.full(annotations["boxes3d"].shape[0], True)

        if self.point_based_filter:
            boxes3d_num_lidar_pts = annotations["boxes3d_num_lidar_pts"]
            boxes3d_num_radar_pts = annotations["boxes3d_num_radar_pts"]
            valid_mask = np.logical_and(
                (boxes3d_num_lidar_pts + boxes3d_num_radar_pts) > 0, valid_mask
            )

        if self.distance_based_filter:
            raise NotImplementedError(
                "Distance based filter not implemented yet"
            )

        boxes3d = annotations["boxes3d"][valid_mask]
        boxes3d_classes = annotations["boxes3d_classes"][valid_mask]
        boxes3d_attributes = annotations["boxes3d_attributes"][valid_mask]
        boxes3d_track_ids = annotations["boxes3d_track_ids"][valid_mask]
        boxes3d_velocities = annotations["boxes3d_velocities"][valid_mask]

        return (
            valid_mask,
            boxes3d,
            boxes3d_classes,
            boxes3d_attributes,
            boxes3d_track_ids,
            boxes3d_velocities,
        )

    def __len__(self) -> int:
        """Length."""
        return len(self.samples)  # type: ignore

    def __getitem__(self, idx: int) -> DictData:
        """Get single sample.

        Args:
            idx (int): Index of sample.

        Returns:
            DictData: sample at index in Vis4D input format.
        """
        sample = self.samples[idx]
        data_dict: DictData = {}

        # TODO: add support for keys_to_load for lidar data
        if "LIDAR_TOP" in self.sensors or K.depth_maps in self.keys_to_load:
            lidar_data = sample["LIDAR_TOP"]

            points_bytes = self.data_backend.get(lidar_data["lidar_path"])
            points = np.frombuffer(points_bytes, dtype=np.float32)
            points = points.reshape(-1, 5)[:, :3]

        if K.depth_maps in self.keys_to_load:
            lidar_to_global = lidar_data["extrinsics"]

            # load LiDAR frame
            if "LIDAR_TOP" in self.sensors:
                (
                    _,
                    boxes3d,
                    boxes3d_classes,
                    boxes3d_attributes,
                    boxes3d_track_ids,
                    boxes3d_velocities,
                ) = self._filter_boxes(lidar_data["annotations"])

                data_dict["LIDAR_TOP"] = {
                    "token": sample["token"],
                    K.points3d: points,
                    K.extrinsics: lidar_data["extrinsics"],
                    K.timestamp: lidar_data["timestamp"],
                    K.frame_ids: sample["frame_ids"],
                    K.axis_mode: AxisMode.ROS,
                    K.boxes3d: boxes3d,
                    K.boxes3d_classes: boxes3d_classes,
                    K.boxes3d_track_ids: boxes3d_track_ids,
                    K.boxes3d_velocities: boxes3d_velocities,
                    "attributes": boxes3d_attributes,
                }

        # load camera frame
        for cam in NuScenes._CAMERAS:
            if cam in self.sensors:
                cam_data = sample[cam]

                data_dict[cam] = {
                    "token": sample["token"],
                    K.frame_ids: sample["frame_ids"],
                    K.timestamp: cam_data["timestamp"],
                }

                if K.images in self.keys_to_load:
                    im_bytes = self.data_backend.get(cam_data["image_path"])
                    image = np.ascontiguousarray(
                        im_decode(im_bytes), dtype=np.float32
                    )[None]

                    data_dict[cam][K.images] = image
                    data_dict[cam][K.input_hw] = cam_data["image_hw"]
                    data_dict[cam][K.sample_names] = cam_data["sample_name"]
                    data_dict[cam][K.intrinsics] = cam_data["intrinsics"]

                if K.original_images in self.keys_to_load:
                    data_dict[cam][K.original_images] = image
                    data_dict[cam][K.original_hw] = cam_data["image_hw"]

                if (
                    K.boxes3d in self.keys_to_load
                    or K.boxes2d in self.keys_to_load
                ):
                    (
                        mask,
                        boxes3d,
                        boxes3d_classes,
                        boxes3d_attributes,
                        boxes3d_track_ids,
                        boxes3d_velocities,
                    ) = self._filter_boxes(cam_data["annotations"])

                    if K.boxes3d in self.keys_to_load:
                        data_dict[cam][K.boxes3d] = boxes3d
                        data_dict[cam][K.boxes3d_classes] = boxes3d_classes
                        data_dict[cam][K.boxes3d_track_ids] = boxes3d_track_ids
                        data_dict[cam][
                            K.boxes3d_velocities
                        ] = boxes3d_velocities
                        data_dict[cam]["attributes"] = boxes3d_attributes
                        data_dict[cam][K.extrinsics] = cam_data["extrinsics"]
                        data_dict[cam][K.axis_mode] = AxisMode.OPENCV

                    if K.boxes2d in self.keys_to_load:
                        boxes2d = cam_data["annotations"]["boxes2d"][mask]

                        data_dict[cam][K.boxes2d] = boxes2d
                        data_dict[cam][K.boxes2d_classes] = boxes3d_classes
                        data_dict[cam][K.boxes2d_track_ids] = boxes3d_track_ids

                if K.depth_maps in self.keys_to_load:
                    depth_maps = self._load_depth_map(
                        points,
                        lidar_to_global,
                        cam_data["extrinsics"],
                        cam_data["intrinsics"],
                        cam_data["image_hw"],
                    )

                    data_dict[cam][K.depth_maps] = depth_maps

        return data_dict
