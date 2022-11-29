"""NuScenes multi-sensor video dataset."""
from __future__ import annotations

import copy
import os
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor

from vis4d.common.imports import NUSCENES_AVAILABLE
from vis4d.common.typing import DictStrAny
from vis4d.data.const import AxisMode, CommonKeys
from vis4d.data.datasets import Dataset, VideoMixin
from vis4d.data.datasets.util import CacheMappingMixin, im_decode
from vis4d.data.io import DataBackend, FileBackend
from vis4d.data.typing import DictData
from vis4d.op.box.box3d import boxes3d_to_corners
from vis4d.op.geometry.projection import project_points
from vis4d.op.geometry.transform import inverse_rigid_transform

if NUSCENES_AVAILABLE:
    from nuscenes import NuScenes as NuScenesDevkit
    from nuscenes.eval.detection.utils import category_to_detection_name
    from nuscenes.utils.data_classes import Box, Quaternion
    from nuscenes.utils.geometry_utils import transform_matrix
    from nuscenes.utils.splits import create_splits_scenes


nuscenes_track_map = {
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


def _get_extrinsics(
    ego_pose: DictStrAny, car_from_sensor: DictStrAny
) -> torch.Tensor:
    """Convert NuScenes ego pose / sensor_to_car to global extrinsics."""
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
    extrinsics = np.dot(global_from_car, car_from_sensor_)
    return torch.tensor(extrinsics, dtype=torch.float32)


class NuScenes(Dataset, CacheMappingMixin, VideoMixin):
    """NuScenes multi-sensor video dataset.

    This dataset loads both LiDAR and camera inputs from the NuScenes dataset
    into the Vis4D expected format for multi-sensor, video datasets.
    """

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
        version: str = "v1.0-trainval",
        split: str = "train",
        include_non_key: bool = False,
        data_backend: DataBackend | None = None,
    ) -> None:
        """Init.

        Args:
            data_root (str): Root directory of nuscenes data in original
                format.
            version (str, optional): Version of the data to load. Defaults to
                "v1.0-trainval".
            split (str, optional): Split of the data to load. Defaults to
                "train".
            include_non_key (bool, optional): Whether to include non-key
                samples. Note that Non-key samples do not have annotations.
                Defaults to False.
            data_backend (Optional[DataBackend], optional): Which data backend
                to use to load the NuScenes data. Defaults to None.
        """
        super().__init__()
        self.data_backend = (
            FileBackend() if data_backend is None else data_backend
        )
        self.data_root = data_root
        assert version in ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
        self.version = version
        if "mini" in version:
            assert split in [
                "mini_train",
                "mini_val",
            ], f"Invalid split for NuScenes {version}!"
        elif "test" in version:
            split = "test"
        else:
            assert split in [
                "train",
                "val",
            ], f"Invalid split for NuScenes {version}!"  # TODO improve error msg for mini version

        self.split = split
        self.include_non_key = include_non_key

        self.data = NuScenesDevkit(
            version=self.version, dataroot=self.data_root, verbose=False
        )
        self.samples = self._load_mapping(
            self._generate_data_mapping,
            os.path.join(
                self.data_root,
                self.version,
                self.split + ".pkl",
            ),
        )
        self.instance_tokens = []

    @property
    def video_to_indices(self) -> dict[str, list[int]]:
        """Mapping from video name to list of dataset indices.

        Required by VideoMixin. Used to split sequences across GPUs and for
        reference view sampling.

        Returns:
            Dict[str, List[int]]: video to indices.
        """
        video_mapping = defaultdict(list)
        for i, sample in enumerate(self.samples):
            video_mapping[sample["scene_token"]].append(i)
        return video_mapping

    def _generate_data_mapping(self) -> list[DictStrAny]:
        """Generate mapping to cache.

        Returns:
            List[DictStrAny]: List of items required to load for a single
                dataset sample.
        """
        samples = []
        scene_names_per_split = create_splits_scenes()

        for record in self.data.scene:
            scene_name = record["name"]
            if scene_name not in scene_names_per_split[self.split]:
                continue

            sample_token = record["first_sample_token"]
            while sample_token:
                sample = self.data.get("sample", sample_token)
                sample_token = sample["next"]
                samples.append(sample)  # TODO non-key frames
        return samples

    def __len__(self) -> int:
        """Length."""
        return len(self.samples)

    def _load_lidar_data(
        self, lidar_data: DictStrAny, ego_pose: DictStrAny
    ) -> tuple[Tensor, Tensor, str]:
        """Load LiDAR data.

        Args:
            lidar_data (DictStrAny): NuScenes format LiDAR data.
            ego_pose (DictStrAny): Ego vehicle pose in NuScenes format.

        Returns:
            tuple[Tensor, Tensor, str]: Pointcloud, extrinsics, timestamp.
        """
        calibration_lidar = self.data.get(
            "calibrated_sensor", lidar_data["calibrated_sensor_token"]
        )
        timestamp = lidar_data["timestamp"]
        lidar_filepath = lidar_data["filename"]
        points_bytes = self.data_backend.get(
            os.path.join(self.data_root, lidar_filepath)
        )
        points = torch.frombuffer(points_bytes, dtype=torch.float32)
        points = points.view(-1, 5)[:, :3]
        extrinsics = _get_extrinsics(ego_pose, calibration_lidar)
        return points, extrinsics, timestamp

    def _load_cam_data(
        self, cam_token: str
    ) -> tuple[Tensor, Tensor, Tensor, str]:
        """Load camera data.

        Args:
            cam_token (str): Token of camera.

        Returns:
            tuple[Tensor, Tensor, Tensor, str]: Image, intrinscs, extrinsics,
                timestamp.
        """
        cam_data = self.data.get("sample_data", cam_token)
        ego_pose_cam = self.data.get("ego_pose", cam_data["ego_pose_token"])
        timestamp = cam_data["timestamp"]
        cam_filepath = cam_data["filename"]
        calibration_cam = self.data.get(
            "calibrated_sensor", cam_data["calibrated_sensor_token"]
        )
        im_bytes = self.data_backend.get(
            os.path.join(self.data_root, cam_filepath)
        )
        image = (
            torch.from_numpy(im_decode(im_bytes))
            .to(torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        extrinsics = _get_extrinsics(ego_pose_cam, calibration_cam)
        intrinsics = torch.tensor(
            calibration_cam["camera_intrinsic"], dtype=torch.float32
        )
        return image, intrinsics, extrinsics, timestamp

    def _get_track_id(self, box: Box) -> int:
        """Get track id for a NuScenes box annotation."""
        instance_token = self.data.get("sample_annotation", box.token)[
            "instance_token"
        ]
        if not instance_token in self.instance_tokens:
            self.instance_tokens.append(instance_token)
        return self.instance_tokens.index(instance_token)

    def _load_boxes3d(
        self,
        list_boxes: list[Box],
        sensor_extrinsics: Tensor | None = None,
        axis_mode: AxisMode = AxisMode.ROS,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Load 3D bounding boxes.

        Args:
            list_boxes (List[Box]): List of boxes in NuScenes format.
            sensor_extrinsics (Optional[torch.Tensor], optional): Extrinsics
                of current sensor. Defaults to None.
            axis_mode (AxisMode, optional): Axis mode of current sensor.
                Defaults to AxisMode.ROS.

        Returns:
            tuple[Tensor, Tensor, Tensor]: 3D boxes, classes and track ids in
                Vis4D format.
        """
        boxes = copy.deepcopy(list_boxes)
        if sensor_extrinsics is not None:
            inverse_extrinsics = inverse_rigid_transform(sensor_extrinsics)
            translation_sensor = inverse_extrinsics[:3, 3].numpy()
            rotation_sensor = Quaternion._from_matrix(
                inverse_extrinsics[:3, :3].numpy(), atol=1e-5
            )
            for box in boxes:
                box.rotate(rotation_sensor)
                box.translate(translation_sensor)

        boxes_tensor, boxes_classes, boxes_track_ids = (
            torch.empty((1, 10), dtype=torch.float32),
            torch.empty((1,), dtype=torch.long),
            torch.empty((1,), dtype=torch.long),
        )
        for box in boxes:
            box_class = category_to_detection_name(box.name)
            if box_class is None:
                continue
            boxes_classes = torch.cat(
                [
                    boxes_classes,
                    torch.tensor(
                        [nuscenes_track_map[box_class]], dtype=torch.long
                    ),
                ]
            )
            boxes_track_ids = torch.cat(
                [
                    boxes_track_ids,
                    torch.tensor([self._get_track_id(box)], dtype=torch.long),
                ]
            )
            if axis_mode == AxisMode.OPENCV:
                # 3D Boxes are aligned with LiDAR coord system (ROS convention)
                # hence we need to extract vertical rotation and switch axis
                v = np.dot(
                    box.orientation.rotation_matrix, np.array([1, 0, 0])
                )
                yaw = -np.arctan2(v[2], v[0])
                box.orientation = Quaternion(angle=yaw, axis=[0, 1, 0])

            box_params = torch.tensor(
                [[*box.center, *box.wlh, *box.orientation]],
                dtype=torch.float32,
            )
            boxes_tensor = torch.cat([boxes_tensor, box_params])
        return boxes_tensor[1:], boxes_classes[1:], boxes_track_ids[1:]

    def _load_boxes2d(
        self,
        boxes3d: torch.Tensor,
        intrinsics: torch.Tensor,
        im_hw: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        """Load 2D bounding boxes.

        Args:
            boxes3d (torch.Tensor): Tensor of 3D boxes in camera frame.
            intrinsics (torch.Tensor): Camera intrinsics
            im_hw (Tuple[int, int]): Image dimensions.

        Returns:
            tuple[Tensor, Tensor]: Mask of 3D bounding boxes successfully
                projecting to the image, the corresponding 2D bounding boxes.
        """
        box_corners = boxes3d_to_corners(boxes3d, AxisMode.OPENCV)
        points = project_points(box_corners.view(-1, 3), intrinsics).view(
            -1, 8, 2
        )
        mask = (points[..., 0] >= 0) * (points[..., 0] < im_hw[1]) * (
            points[..., 1] >= 0
        ) * (points[..., 1] < im_hw[0]) * box_corners[..., 2] > 0.0
        mask = mask.any(dim=-1)
        points = points[mask]
        points[..., 0] = points[..., 0].clamp(min=0, max=im_hw[1])
        points[..., 1] = points[..., 1].clamp(min=0, max=im_hw[0])
        boxes2d = torch.stack(
            (
                points[..., 0].min(dim=1)[0],
                points[..., 1].min(dim=1)[0],
                points[..., 0].max(dim=1)[0],
                points[..., 1].max(dim=1)[0],
            ),
            dim=-1,
        )
        return mask, boxes2d

    def __getitem__(self, idx: int) -> DictData:
        """Get single sample.

        Args:
            idx (int): Index of sample.

        Returns:
            DictData: sample at index in Vis4D input format.
        """
        sample = self.samples[idx]
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = self.data.get("sample_data", lidar_token)

        ego_pose = self.data.get("ego_pose", lidar_data["ego_pose_token"])
        boxes = self.data.get_boxes(lidar_token)

        # load LiDAR frame
        data_dict: DictData = {}
        if "LIDAR_TOP" in self._SENSORS:
            points, extrinsics, timestamp = self._load_lidar_data(
                lidar_data, ego_pose
            )
            # TODO add track ids
            boxes3d, boxes3d_classes, boxes3d_track_ids = self._load_boxes3d(
                boxes, extrinsics
            )
            data_dict["LIDAR_TOP"] = {
                CommonKeys.points3d: points,
                CommonKeys.extrinsics: extrinsics,
                CommonKeys.timestamp: timestamp,
                CommonKeys.axis_mode: AxisMode.ROS,
                CommonKeys.boxes3d: boxes3d,
                CommonKeys.boxes3d_classes: boxes3d_classes,
                CommonKeys.boxes3d_track_ids: boxes3d_track_ids,
            }

        # load camera frames
        for cam in NuScenes._CAMERAS:
            if cam in self._SENSORS:
                cam_token = sample["data"][cam]
                image, intrinsics, extrinsics, timestamp = self._load_cam_data(
                    cam_token
                )
                image_hw = image.size(2), image.size(3)
                (
                    boxes3d,
                    boxes3d_classes,
                    boxes3d_track_ids,
                ) = self._load_boxes3d(boxes, extrinsics, AxisMode.OPENCV)

                mask, boxes2d = self._load_boxes2d(
                    boxes3d, intrinsics, image_hw
                )
                data_dict[cam] = {
                    CommonKeys.images: image,
                    CommonKeys.original_hw: image_hw,
                    CommonKeys.input_hw: image_hw,
                    CommonKeys.intrinsics: intrinsics,
                    CommonKeys.extrinsics: extrinsics,
                    CommonKeys.timestamp: timestamp,
                    CommonKeys.axis_mode: AxisMode.OPENCV,
                    CommonKeys.boxes2d: boxes2d,
                    CommonKeys.boxes2d_classes: boxes3d_classes[mask],
                    CommonKeys.boxes2d_track_ids: boxes3d_track_ids[mask],
                    CommonKeys.boxes3d: boxes3d[mask],
                    CommonKeys.boxes3d_classes: boxes3d_classes[mask],
                    CommonKeys.boxes3d_track_ids: boxes3d_track_ids[mask],
                }

        # TODO add RADAR, Map data
        return data_dict
