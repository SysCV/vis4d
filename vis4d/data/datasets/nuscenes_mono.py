"""NuScenes monocular dataset."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from vis4d.common.imports import NUSCENES_AVAILABLE
from vis4d.common.logging import rank_zero_info
from vis4d.common.time import Timer
from vis4d.common.typing import ArgsType, DictStrAny
from vis4d.data.const import AxisMode
from vis4d.data.const import CommonKeys as K
from vis4d.data.typing import DictData

from .nuscenes import NuScenes, nuscenes_class_map
from .util import im_decode, print_class_histogram

if NUSCENES_AVAILABLE:
    from nuscenes import NuScenes as NuScenesDevkit
    from nuscenes.utils.splits import create_splits_scenes
else:
    raise ImportError("nusenes-devkit is not available.")


class NuScenesMono(NuScenes):
    """NuScenes monocular dataset."""

    def __init__(self, *args: ArgsType, **kwargs: ArgsType) -> None:
        """Initialize the dataset."""
        super().__init__(*args, **kwargs)

    # Needed for CBGS
    def get_cat_ids(self, idx: int) -> list[int]:
        """Return the samples."""
        return self.samples[idx]["CAM"]["annotations"]["boxes3d_classes"]

    def _filter_data(self, data: list[DictStrAny]) -> list[DictStrAny]:
        """Remove empty samples."""
        samples = []
        frequencies = {cat: 0 for cat in nuscenes_class_map}
        inv_nuscenes_class_map = {v: k for k, v in nuscenes_class_map.items()}

        t = Timer()
        for sample in data:
            (
                mask,
                boxes3d,
                boxes3d_classes,
                boxes3d_attributes,
                boxes3d_track_ids,
                boxes3d_velocities,
            ) = self._filter_boxes(sample["CAM"]["annotations"])

            sample["CAM"]["annotations"]["boxes3d"] = boxes3d
            sample["CAM"]["annotations"]["boxes3d_classes"] = boxes3d_classes
            sample["CAM"]["annotations"][
                "boxes3d_attributes"
            ] = boxes3d_attributes
            sample["CAM"]["annotations"][
                "boxes3d_track_ids"
            ] = boxes3d_track_ids
            sample["CAM"]["annotations"][
                "boxes3d_velocities"
            ] = boxes3d_velocities
            sample["CAM"]["annotations"]["boxes2d"] = sample["CAM"][
                "annotations"
            ]["boxes2d"][mask]

            for box3d_class in boxes3d_classes:
                frequencies[inv_nuscenes_class_map[box3d_class]] += 1

            if self.skip_empty_samples:
                if len(sample["CAM"]["annotations"]["boxes3d"]) > 0:
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
            frame_ids = 0
            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = data.get("sample", sample_token)

                # LIDAR data
                lidar_token = sample["data"]["LIDAR_TOP"]

                lidar_data = self._load_lidar_data(data, lidar_token)
                lidar_data["annotations"] = self._load_annotations(
                    data,
                    lidar_data["extrinsics"],
                    sample["anns"],
                    instance_tokens,
                )

                # TODO add RADAR, Map data

                # Get the sample data for each camera
                for cam in self.CAMERAS:
                    frame: DictStrAny = {}
                    frame["scene_name"] = f"{scene_name}_{cam}"
                    frame["token"] = sample["token"]
                    frame["frame_ids"] = frame_ids

                    frame["LIDAR_TOP"] = lidar_data

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
            K.sequence_names: sample["scene_name"],
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

        if K.boxes3d in self.keys_to_load or K.boxes2d in self.keys_to_load:
            if K.boxes3d in self.keys_to_load:
                data_dict[K.boxes3d] = sample["CAM"]["annotations"]["boxes3d"]
                data_dict[K.boxes3d_classes] = sample["CAM"]["annotations"][
                    "boxes3d_classes"
                ]
                data_dict[K.boxes3d_track_ids] = sample["CAM"]["annotations"][
                    "boxes3d_track_ids"
                ]
                data_dict[K.boxes3d_velocities] = sample["CAM"]["annotations"][
                    "boxes3d_velocities"
                ]
                data_dict["attributes"] = sample["CAM"]["annotations"][
                    "boxes3d_attributes"
                ]
                data_dict[K.extrinsics] = sample["CAM"]["extrinsics"]
                data_dict[K.axis_mode] = AxisMode.OPENCV

            if K.boxes2d in self.keys_to_load:
                data_dict[K.boxes2d] = sample["CAM"]["annotations"]["boxes2d"]
                data_dict[K.boxes2d_classes] = data_dict[K.boxes3d_classes]
                data_dict[K.boxes2d_track_ids] = data_dict[K.boxes3d_track_ids]

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
