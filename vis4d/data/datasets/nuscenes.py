"""NuScenes multi-sensor video dataset."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from vis4d.common.imports import NUSCENES_AVAILABLE
from vis4d.common.logging import rank_zero_info
from vis4d.common.time import Timer
from vis4d.common.typing import (
    ArgsType,
    DictStrAny,
    NDArrayBool,
    NDArrayF32,
    NDArrayI64,
)
from vis4d.data.const import AxisMode
from vis4d.data.const import CommonKeys as K
from vis4d.data.typing import DictData
from vis4d.op.geometry.projection import generate_depth_map
from vis4d.op.geometry.transform import (
    inverse_rigid_transform,
    transform_points,
)

from .base import VideoDataset, VideoMapping
from .util import CacheMappingMixin, im_decode, print_class_histogram

if NUSCENES_AVAILABLE:
    from nuscenes import NuScenes as NuScenesDevkit
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.eval.common.utils import quaternion_yaw
    from nuscenes.eval.detection.utils import category_to_detection_name
    from nuscenes.scripts.export_2d_annotations_as_json import (
        post_process_coords,
    )
    from nuscenes.utils.data_classes import Quaternion
    from nuscenes.utils.geometry_utils import (
        box_in_image,
        transform_matrix,
        view_points,
    )
    from nuscenes.utils.splits import create_splits_scenes
else:
    raise ImportError("nusenes-devkit is not available.")

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

    SENSORS = [
        "LIDAR_TOP",
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    CAMERAS = [
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
            "LIDAR_TOP",
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ),
        version: str = "v1.0-trainval",
        split: str = "train",
        max_sweeps: int = 10,
        skip_empty_samples: bool = False,
        point_based_filter: bool = False,
        distance_based_filter: bool = False,
        cache_as_binary: bool = False,
        cached_file_path: str | None = None,
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class.

        Args:
            data_root (str): Root directory of nuscenes data in original
                format.
            keys_to_load (tuple[str, ...]): Keys to load from the dataset.
                Defaults to (K.images, K.boxes2d, K.boxes3d).
            sensors (Sequence[str, ...]): Which sensor to load. Defaults
                to ("LIDAR_TOP", "CAM_FRONT", "CAM_FRONT_LEFT",
                "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT").
            version (str, optional): Version of the data to load. Defaults to
                "v1.0-trainval".
            split (str, optional): Split of the data to load. Defaults to
                "train".
            max_sweeps (int, optional): Maximum number of sweeps for a single
                key-frame to load. Defaults to 10.
            skip_empty_samples (bool, optional): Whether to skip samples
                without annotations. Defaults to False.
            point_based_filter (bool, optional): Whether to filter out
                samples based on the number of points in the point cloud.
                Defaults to False.
            distance_based_filter (bool, optional): Whether to filter out
                samples based on the distance of the object from the ego
                vehicle. Defaults to False.
            cache_as_binary (bool): Whether to cache the dataset as binary.
                Default: False.
            cached_file_path (str | None): Path to a cached file. If cached
                file exist then it will load it instead of generating the data
                mapping. Default: None.
        """
        super().__init__(**kwargs)
        self.data_root = data_root
        self.keys_to_load = keys_to_load
        self.sensors = sensors
        self._check_version_and_split(version, split)
        self.max_sweeps = max_sweeps
        self.skip_empty_samples = skip_empty_samples

        self.point_based_filter = point_based_filter
        self.distance_based_filter = distance_based_filter

        # Load annotations
        self.samples, self.original_len = self._load_mapping(
            self._generate_data_mapping,
            self._filter_data,
            cache_as_binary=cache_as_binary,
            cached_file_path=cached_file_path,
        )

        # Generate video mapping
        self.video_mapping = self._generate_video_mapping()

    # Needed for CBGS
    def get_cat_ids(self, idx: int) -> list[int]:
        """Return the samples."""
        return self.samples[idx]["LIDAR_TOP"]["annotations"]["boxes3d_classes"]

    def _filter_data(self, data: list[DictStrAny]) -> list[DictStrAny]:
        """Remove empty samples."""
        if self.split == "test":
            return data

        samples = []
        frequencies = {cat: 0 for cat in nuscenes_class_map}
        inv_nuscenes_class_map = {v: k for k, v in nuscenes_class_map.items()}

        t = Timer()
        for sample in data:
            (
                _,
                boxes3d,
                boxes3d_classes,
                boxes3d_attributes,
                boxes3d_track_ids,
                boxes3d_velocities,
            ) = self._filter_boxes(sample["LIDAR_TOP"]["annotations"])

            sample["LIDAR_TOP"]["annotations"]["boxes3d"] = boxes3d
            sample["LIDAR_TOP"]["annotations"][
                "boxes3d_classes"
            ] = boxes3d_classes
            sample["LIDAR_TOP"]["annotations"][
                "boxes3d_attributes"
            ] = boxes3d_attributes
            sample["LIDAR_TOP"]["annotations"][
                "boxes3d_track_ids"
            ] = boxes3d_track_ids
            sample["LIDAR_TOP"]["annotations"][
                "boxes3d_velocities"
            ] = boxes3d_velocities

            for box3d_class in boxes3d_classes:
                frequencies[inv_nuscenes_class_map[box3d_class]] += 1

            for cam in NuScenes.CAMERAS:
                (
                    mask,
                    boxes3d,
                    boxes3d_classes,
                    boxes3d_attributes,
                    boxes3d_track_ids,
                    boxes3d_velocities,
                ) = self._filter_boxes(sample[cam]["annotations"])

                sample[cam]["annotations"]["boxes3d"] = boxes3d
                sample[cam]["annotations"]["boxes3d_classes"] = boxes3d_classes
                sample[cam]["annotations"][
                    "boxes3d_attributes"
                ] = boxes3d_attributes
                sample[cam]["annotations"][
                    "boxes3d_track_ids"
                ] = boxes3d_track_ids
                sample[cam]["annotations"][
                    "boxes3d_velocities"
                ] = boxes3d_velocities
                sample[cam]["annotations"]["boxes2d"] = sample[cam][
                    "annotations"
                ]["boxes2d"][mask]

            if self.skip_empty_samples:
                if len(sample["LIDAR_TOP"]["annotations"]["boxes3d"]) > 0:
                    samples.append(sample)
            else:
                samples.append(sample)

        rank_zero_info(
            f"Preprocessing {len(data)} frames takes {t.time():.2f}"
            " seconds."
        )

        print_class_histogram(frequencies)

        if self.skip_empty_samples:
            rank_zero_info(
                f"Filtered {len(data) - len(samples)} empty frames."
            )

        return samples

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

    def _generate_video_mapping(self) -> VideoMapping:
        """Group dataset sample indices by their associated video ID.

        The sample index is an integer while video IDs are string.

        Returns:
            VideoMapping: Mapping of video IDs to sample indices and frame IDs.
        """
        video_to_indices: dict[str, list[int]] = defaultdict(list)
        video_to_frame_ids: dict[str, list[int]] = defaultdict(list)
        for i, sample in enumerate(self.samples):  # type: ignore
            seq = sample["scene_name"]
            video_to_indices[seq].append(i)
            video_to_frame_ids[seq].append(sample["frame_ids"])

        return self._sort_video_mapping(
            {
                "video_to_indices": video_to_indices,
                "video_to_frame_ids": video_to_frame_ids,
            }
        )

    def _generate_data_mapping(self) -> list[DictStrAny]:
        """Generate data mapping.

        Returns:
            List[DictStrAny]: List of items required to load for a single
                dataset sample.
        """
        data = NuScenesDevkit(
            version=self.version, dataroot=self.data_root, verbose=False
        )

        can_bus_data = NuScenesCanBus(dataroot=self.data_root)

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

                sd_rec = data.get("sample_data", sample["data"]["LIDAR_TOP"])

                # Can bus data
                can_bus = self._load_can_bus_data(
                    scene_name, can_bus_data, sample["timestamp"]
                )

                pose_record = data.get("ego_pose", sd_rec["ego_pose_token"])
                rotation = Quaternion(pose_record["rotation"])
                translation = pose_record["translation"]

                can_bus[:3] = translation
                can_bus[3:7] = rotation
                patch_angle = quaternion_yaw(rotation) / np.pi * 180
                patch_angle += 360 if patch_angle < 0 else 0
                can_bus[-2] = patch_angle / 180 * np.pi
                can_bus[-1] = patch_angle

                frame["can_bus"] = can_bus

                # LIDAR data
                lidar_token = sample["data"]["LIDAR_TOP"]

                frame["LIDAR_TOP"] = self._load_lidar_data(data, lidar_token)

                if self.split != "test":
                    frame["LIDAR_TOP"]["annotations"] = self._load_annotations(
                        data,
                        frame["LIDAR_TOP"]["extrinsics"],
                        sample["anns"],
                        instance_tokens,
                        axis_mode=AxisMode.LIDAR,
                    )

                # obtain sweeps for a single key-frame
                sweeps: list[DictStrAny] = []
                while len(sweeps) < self.max_sweeps:
                    if sd_rec["prev"] != "":
                        sweep = self._load_lidar_data(data, sd_rec["prev"])
                        sweeps.append(sweep)
                        sd_rec = data.get("sample_data", sd_rec["prev"])
                    else:
                        break
                frame["LIDAR_TOP"]["sweeps"] = sweeps

                # Get the sample data for each camera
                for cam in self.CAMERAS:
                    cam_token = sample["data"][cam]

                    frame[cam] = self._load_cam_data(data, cam_token)

                    if self.split != "test":
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

                # TODO add RADAR, Map

                frames.append(frame)

                sample_token = sample["next"]
                frame_ids += 1

        return frames

    def _load_can_bus_data(
        self,
        scene_name: str,
        can_bus_data: NuScenesCanBus,
        sample_timestamp: int,
    ) -> list[float]:
        """Load can bus data."""
        try:
            pose_list = can_bus_data.get_messages(scene_name, "pose")
        except:  # pylint: disable=bare-except
            # server scenes do not have can bus information.
            return [0.0] * 18

        # during each scene, the first timestamp of can_bus may be large than
        # the first sample's timestamp
        can_bus = []
        last_pose = pose_list[0]
        for pose in pose_list:
            if pose["utime"] > sample_timestamp:
                break
            last_pose = pose

        last_pose.pop("utime")
        pos = last_pose.pop("pos")
        rotation = last_pose.pop("orientation")
        can_bus.extend(pos)
        can_bus.extend(rotation)

        # 16 elements
        for key in last_pose.keys():
            can_bus.extend(last_pose[key])
        can_bus.extend([0.0, 0.0])

        return can_bus

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
            lidar_data["filename"].split("/")[-1].replace(".pcd.bin", "")
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
            data (NuScenesDevkit): NuScenes toolkit.
            cam_token (str): Camera token.

        Returns:
            DictStrAny: Camera data containing the sample name, image path,
                image height and width, intrinsics, extrinsics, and
                timestamp.
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

            if export_2d_annotations:
                assert (
                    image_hw is not None
                ), "Image height and width must be provided."
                if not box_in_image(
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

            # Get 3D box yaw. Use extrinsic rotation to align with PyTorch3D.
            if axis_mode == AxisMode.OPENCV:
                yaw = -box3d.orientation.yaw_pitch_roll[0]
                x, y, z, w = R.from_euler("XYZ", [0, yaw, 0]).as_quat()
            else:
                yaw = box3d.orientation.yaw_pitch_roll[0]
                x, y, z, w = R.from_euler("XYZ", [0, 0, yaw]).as_quat()

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

    def _accumulate_sweeps(
        self,
        points: NDArrayF32,
        lidar2global: NDArrayF32,
        sweeps: list[DictStrAny],
    ) -> NDArrayF32:
        """Accumulate LiDAR sweeps."""
        if len(sweeps) == 0:
            return points

        global2lidar = inverse_rigid_transform(torch.from_numpy(lidar2global))

        points_sweeps = [torch.from_numpy(points)]
        for sweep in sweeps:
            points_bytes = self.data_backend.get(sweep["lidar_path"])
            lidar_points = np.frombuffer(
                bytearray(points_bytes), dtype=np.float32
            )
            lidar_points = lidar_points.reshape(-1, 5)[:, :3]

            # Transform LiDAR points to global frame
            global_lidar_points = transform_points(
                torch.from_numpy(lidar_points),
                torch.from_numpy(sweep["extrinsics"]),
            )

            # Transform LiDAR points to current LiDAR frame
            current_lidar_points = transform_points(
                global_lidar_points, global2lidar
            )

            points_sweeps.append(current_lidar_points)

        return torch.cat(points_sweeps).numpy()

    def _load_depth_map(
        self,
        points_lidar: NDArrayF32,
        lidar2global: NDArrayF32,
        cam2global: NDArrayF32,
        intrinsics: NDArrayF32,
        image_hw: tuple[int, int],
    ) -> NDArrayF32:
        """Load depth map.

        Args:
            points_lidar (NDArrayF32): LiDAR points.
            lidar2global (NDArrayF32): LiDAR to global extrinsics.
            cam2global (NDArrayF32): Camera to global extrinsics.
            intrinsics (NDArrayF32): Camera intrinsic matrix.
            image_hw (tuple[int, int]): Image height and width.

        Returns:
            NDArrayF32: Depth map.
        """
        cam2global_ = torch.from_numpy(cam2global)
        lidar2global_ = torch.from_numpy(lidar2global)
        intrinsics_ = torch.from_numpy(intrinsics)
        points_lidar_ = torch.from_numpy(np.copy(points_lidar))

        lidar2cam = torch.matmul(torch.inverse(cam2global_), lidar2global_)
        cam2img = torch.eye(4, 4)
        cam2img[:3, :3] = intrinsics_
        points_cam = points_lidar_[:, :3] @ (lidar2cam[:3, :3].T) + lidar2cam[
            :3, 3
        ].unsqueeze(0)

        depth_map = generate_depth_map(points_cam, intrinsics_, image_hw)
        return depth_map.numpy()

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
        return len(self.samples)

    def __getitem__(self, idx: int) -> DictData:
        """Get single sample.

        Args:
            idx (int): Index of sample.

        Returns:
            DictData: sample at index in Vis4D input format.
        """
        sample = self.samples[idx]
        data_dict: DictData = {}

        # metadata
        data_dict["token"] = sample["token"]
        data_dict[K.frame_ids] = sample["frame_ids"]
        data_dict[K.sequence_names] = sample["scene_name"]
        data_dict["can_bus"] = sample["can_bus"]

        if "LIDAR_TOP" in self.sensors:
            lidar_data = sample["LIDAR_TOP"]

            # load LiDAR frame
            data_dict["LIDAR_TOP"] = {
                K.sample_names: lidar_data["sample_name"],
                K.timestamp: lidar_data["timestamp"],
                K.extrinsics: lidar_data["extrinsics"],
                K.axis_mode: AxisMode.LIDAR,
            }

            if (
                K.points3d in self.keys_to_load
                or K.depth_maps in self.keys_to_load
            ):
                points_bytes = self.data_backend.get(lidar_data["lidar_path"])
                lidar_points = np.frombuffer(
                    bytearray(points_bytes), dtype=np.float32
                )
                lidar_points = lidar_points.reshape(-1, 5)[:, :3]

                lidar_points = self._accumulate_sweeps(
                    lidar_points,
                    lidar_data["extrinsics"],
                    lidar_data["sweeps"],
                )

            if K.points3d in self.keys_to_load:
                data_dict["LIDAR_TOP"][K.points3d] = lidar_points

            if K.boxes3d in self.keys_to_load:
                data_dict["LIDAR_TOP"][K.boxes3d] = lidar_data["annotations"][
                    "boxes3d"
                ]
                data_dict["LIDAR_TOP"][K.boxes3d_classes] = lidar_data[
                    "annotations"
                ]["boxes3d_classes"]
                data_dict["LIDAR_TOP"][K.boxes3d_track_ids] = lidar_data[
                    "annotations"
                ]["boxes3d_track_ids"]
                data_dict["LIDAR_TOP"][K.boxes3d_velocities] = lidar_data[
                    "annotations"
                ]["boxes3d_velocities"]
                data_dict["LIDAR_TOP"]["attributes"] = lidar_data[
                    "annotations"
                ]["boxes3d_attributes"]

        # load camera frame
        for cam in NuScenes.CAMERAS:
            if cam in self.sensors:
                cam_data = sample[cam]

                data_dict[cam] = {K.timestamp: cam_data["timestamp"]}

                if K.images in self.keys_to_load:
                    im_bytes = self.data_backend.get(cam_data["image_path"])
                    image = np.ascontiguousarray(
                        im_decode(im_bytes, mode=self.image_channel_mode),
                        dtype=np.float32,
                    )[None]

                    data_dict[cam][K.images] = image
                    data_dict[cam][K.input_hw] = cam_data["image_hw"]
                    data_dict[cam][K.sample_names] = cam_data["sample_name"]
                    data_dict[cam][K.intrinsics] = cam_data["intrinsics"]
                    data_dict[cam][K.extrinsics] = cam_data["extrinsics"]
                    data_dict[cam][K.axis_mode] = AxisMode.OPENCV

                if K.original_images in self.keys_to_load:
                    data_dict[cam][K.original_images] = image
                    data_dict[cam][K.original_hw] = cam_data["image_hw"]

                if (
                    K.boxes3d in self.keys_to_load
                    or K.boxes2d in self.keys_to_load
                ):
                    if K.boxes3d in self.keys_to_load:
                        data_dict[cam][K.boxes3d] = cam_data["annotations"][
                            "boxes3d"
                        ]
                        data_dict[cam][K.boxes3d_classes] = cam_data[
                            "annotations"
                        ]["boxes3d_classes"]
                        data_dict[cam][K.boxes3d_track_ids] = cam_data[
                            "annotations"
                        ]["boxes3d_track_ids"]
                        data_dict[cam][K.boxes3d_velocities] = cam_data[
                            "annotations"
                        ]["boxes3d_velocities"]
                        data_dict[cam]["attributes"] = cam_data["annotations"][
                            "boxes3d_attributes"
                        ]

                    if K.boxes2d in self.keys_to_load:
                        boxes2d = cam_data["annotations"]["boxes2d"]

                        data_dict[cam][K.boxes2d] = boxes2d
                        data_dict[cam][K.boxes2d_classes] = data_dict[cam][
                            K.boxes3d_classes
                        ]
                        data_dict[cam][K.boxes2d_track_ids] = data_dict[cam][
                            K.boxes3d_track_ids
                        ]

                if K.depth_maps in self.keys_to_load:
                    depth_maps = self._load_depth_map(
                        lidar_points,
                        lidar_data["extrinsics"],
                        cam_data["extrinsics"],
                        cam_data["intrinsics"],
                        cam_data["image_hw"],
                    )

                    data_dict[cam][K.depth_maps] = depth_maps

        return data_dict
