"""NuScenes monocular dataset."""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from vis4d.common.imports import NUSCENES_AVAILABLE
from vis4d.common.typing import ArgsType, DictStrAny
from vis4d.data.const import AxisMode
from vis4d.data.const import CommonKeys as K
from vis4d.data.typing import DictData

from .nuscenes import NuScenes
from .util import im_decode

if NUSCENES_AVAILABLE:
    from nuscenes import NuScenes as NuScenesDevkit
    from nuscenes.utils.splits import create_splits_scenes


class NuScenesMono(NuScenes):
    """NuScenes monocular dataset."""

    def __init__(self, *args: ArgsType, **kwargs: ArgsType) -> None:
        """Initialize the dataset."""
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        """Concise representation of the dataset."""
        return f"NuScenes Monocular Dataset {self.version} {self.split}"

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
            # Get the sample data for each camera
            for cam in self.CAMERAS:
                frame_ids = 0
                sample_token = scene["first_sample_token"]
                while sample_token:
                    frame: DictStrAny = {}
                    sample = data.get("sample", sample_token)

                    frame["scene_name"] = f"{scene_name}_{cam}"
                    frame["token"] = sample["token"]
                    frame["frame_ids"] = frame_ids

                    lidar_token = sample["data"]["LIDAR_TOP"]

                    frame["LIDAR_TOP"] = self._load_lidar_data(
                        data, lidar_token
                    )
                    frame["LIDAR_TOP"]["annotations"] = self._load_annotations(
                        data,
                        frame["LIDAR_TOP"]["extrinsics"],
                        sample["anns"],
                        instance_tokens,
                    )

                    cam_token = sample["data"][cam]

                    frame["CAM"] = self._load_cam_data(data, cam_token)
                    frame["CAM"]["annotations"] = self._load_annotations(
                        data,
                        frame["CAM"]["extrinsics"],
                        sample["anns"],
                        instance_tokens,
                        axis_mode=AxisMode.OPENCV,
                        export_2d_annotations=True,
                        intrinsics=frame["CAM"]["intrinsics"],
                        image_hw=frame["CAM"]["image_hw"],
                    )

                    # TODO add RADAR, Map data

                    frames.append(frame)

                    sample_token = sample["next"]
                    frame_ids += 1

        return frames

    def __getitem__(self, idx: int) -> DictData:
        """Get single sample.

        Args:
            idx (int): Index of sample.

        Returns:
            DictData: sample at index in Vis4D input format.
        """
        sample = self.samples[idx]
        data_dict: DictData = {}

        if K.depth_maps in self.keys_to_load:
            lidar_data = sample["LIDAR_TOP"]

            points_bytes = self.data_backend.get(lidar_data["lidar_path"])
            points = np.frombuffer(points_bytes, dtype=np.float32)
            points = points.reshape(-1, 5)[:, :3]

        if K.depth_maps in self.keys_to_load:
            lidar_to_global = lidar_data["extrinsics"]

        # load camera frame
        data_dict = {
            "token": sample["token"],
            K.frame_ids: sample["frame_ids"],
            K.timestamp: sample["CAM"]["timestamp"],
        }

        if K.images in self.keys_to_load:
            im_bytes = self.data_backend.get(sample["CAM"]["image_path"])
            image = np.ascontiguousarray(
                im_decode(im_bytes), dtype=np.float32
            )[None]

            data_dict[K.images] = image
            data_dict[K.input_hw] = sample["CAM"]["image_hw"]
            data_dict[K.sample_names] = sample["CAM"]["sample_name"]
            data_dict[K.intrinsics] = sample["CAM"]["intrinsics"]

        if K.original_images in self.keys_to_load:
            data_dict[K.original_images] = image
            data_dict[K.original_hw] = sample["CAM"]["image_hw"]

        (
            mask,
            boxes3d,
            boxes3d_classes,
            boxes3d_attributes,
            boxes3d_track_ids,
            boxes3d_velocities,
        ) = self._filter_boxes(sample["CAM"]["annotations"])

        if K.boxes3d in self.keys_to_load or K.boxes2d in self.keys_to_load:
            (
                mask,
                boxes3d,
                boxes3d_classes,
                boxes3d_attributes,
                boxes3d_track_ids,
                boxes3d_velocities,
            ) = self._filter_boxes(sample["CAM"]["annotations"])

            if K.boxes3d in self.keys_to_load:
                data_dict[K.boxes3d] = boxes3d
                data_dict[K.boxes3d_classes] = boxes3d_classes
                data_dict[K.boxes3d_track_ids] = boxes3d_track_ids
                data_dict[K.boxes3d_velocities] = boxes3d_velocities
                data_dict["attributes"] = boxes3d_attributes
                data_dict[K.extrinsics] = sample["CAM"]["extrinsics"]
                data_dict[K.axis_mode] = AxisMode.OPENCV

            if K.boxes2d in self.keys_to_load:
                boxes2d = sample["CAM"]["annotations"]["boxes2d"][mask]

                data_dict[K.boxes2d] = boxes2d
                data_dict[K.boxes2d_classes] = boxes3d_classes
                data_dict[K.boxes2d_track_ids] = boxes3d_track_ids

        if K.depth_maps in self.keys_to_load:
            depth_maps = self._load_depth_map(
                points,
                lidar_to_global,
                sample["CAM"]["extrinsics"],
                sample["CAM"]["intrinsics"],
                sample["CAM"]["image_hw"],
            )

            data_dict[K.depth_maps] = depth_maps

        return data_dict
