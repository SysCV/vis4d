"""NuScenes trajectory dataset."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

from vis4d.common.imports import NUSCENES_AVAILABLE
from vis4d.common.logging import rank_zero_info
from vis4d.common.typing import DictStrAny, NDArrayF32
from vis4d.data.typing import DictData

from .base import Dataset
from .util import CacheMappingMixin

if NUSCENES_AVAILABLE:
    from nuscenes import NuScenes as NuScenesDevkit
    from nuscenes.eval.detection.utils import category_to_detection_name
    from nuscenes.utils.data_classes import Quaternion
    from nuscenes.utils.splits import create_splits_scenes
else:
    raise ImportError("nusenes-devkit is not available.")


class NuScenesTrajectory(CacheMappingMixin, Dataset):
    """NuScenes Trajectory dataset with given detection results.

    It will generate a trajectory data pair with minimum sequence length. The
    detection results will be matched with the ground truth trajectory
    according to the BEV distance.
    """

    def __init__(
        self,
        detector: str,
        pure_detection: str,
        data_root: str,
        version: str = "v1.0-trainval",
        split: str = "train",
        min_seq_len: int = 10,
        cache_as_binary: bool = False,
        cached_file_path: str | None = None,
    ) -> None:
        """Init dataset.

        Args:
            detector (str): The detector name.
            pure_detection (str): The path to the pure detection results. It
                should be the same format as nuScenes submission format.
            data_root (str): The root path of the dataset.
            version (str, optional): The version of the dataset. Defaults to
                "v1.0-trainval".
            split (str, optional): The split of the dataset. Defaults to
                "train".
            min_seq_len (int, optional): The minimum sequence length of the
                trajectory. Defaults to 10.
            cache_as_binary (bool, optional): Whether to cache the dataset as
                binary. Defaults to False.
            cached_file_path (str | None, optional): The path to the cached
                file. Defaults to None.
        """
        super().__init__()
        self.data_root = data_root
        self.version = version
        self.split = split

        self.detector = detector
        self.min_seq_len = min_seq_len

        self.pure_detection = pure_detection

        # Load trajectories
        self.samples, _ = self._load_mapping(
            self._generate_data_mapping,
            cache_as_binary=cache_as_binary,
            cached_file_path=cached_file_path,
        )
        rank_zero_info(f"Generated {len(self.samples)} trajectories.")

    def __repr__(self) -> str:
        """Concise representation of the dataset."""
        return f"NuScenes Trajectory Data with {self.detector} detection"

    def _match_gt_pred(
        self,
        gt_world: NDArrayF32,
        gt_class: str,
        predictions: list[DictStrAny],
    ) -> tuple[NDArrayF32, bool]:
        """Match gt and pred according to BEV center distance.

        If the distance is less than 2 meters, the prediction will be used
        instead of the ground truth.
        """
        if len(predictions) > 0:
            same_class_preds = [
                pred
                for pred in predictions
                if pred["detection_name"] == gt_class
            ]

            if len(same_class_preds) > 0:
                preds_center = [
                    pred["translation"][:2] for pred in same_class_preds
                ]
                distance_matrix = (
                    cdist(  # pylint: disable=unsubscriptable-object
                        gt_world[:, :2],
                        np.array(preds_center).reshape(-1, 2),
                    )[0]
                )

                if distance_matrix[distance_matrix.argmin()] <= 2:
                    match_pred = same_class_preds[distance_matrix.argmin()]

                    # WLH -> HWL
                    w, l, h = match_pred["size"]
                    dimensions = [h, w, l]
                    yaw = Quaternion(match_pred["rotation"]).yaw_pitch_roll[0]

                    pred_world = np.array(
                        [
                            [
                                *match_pred["translation"],
                                *dimensions,
                                yaw,
                                match_pred["detection_score"],
                            ]
                        ],
                        dtype=np.float32,
                    )

                    return pred_world, False

        return gt_world, True

    def _generate_data_mapping(self) -> list[dict[str, NDArrayF32]]:
        """Generate trajectories predction and groundtruth.

        Trajectories will be generated for each scene. Each trajectory consists
        of [x, y, z, h, w, l, yaw, score] in world coordinate.

        Returns:
            list[dict[str, NDArrayF32]]: The list of trajectories.
        """
        data = NuScenesDevkit(
            version=self.version, dataroot=self.data_root, verbose=False
        )

        scene_names_per_split = create_splits_scenes()

        scenes = [
            scene
            for scene in data.scene
            if scene["name"] in scene_names_per_split[self.split]
        ]

        instance_tokens = []

        with open(self.pure_detection, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        num_gt_boxes = 0
        num_pred_boxes = 0
        total_traj = []
        for scene in tqdm(scenes):
            local_traj: dict[int, dict[str, list[NDArrayF32]]] = {}

            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = data.get("sample", sample_token)

                preds = predictions["results"][sample_token]

                for ann_token in sample["anns"]:
                    ann_info = data.get("sample_annotation", ann_token)
                    box3d_class = category_to_detection_name(
                        ann_info["category_name"]
                    )

                    if box3d_class is None:
                        continue

                    box3d = data.get_box(ann_info["token"])

                    instance_token = data.get(
                        "sample_annotation", box3d.token
                    )["instance_token"]

                    if not instance_token in instance_tokens:
                        instance_tokens.append(instance_token)
                    track_id = instance_tokens.index(instance_token)

                    if track_id not in local_traj:
                        local_traj[track_id] = {"gt": [], "pred": []}

                    # WLH -> HWL
                    w, l, h = box3d.wlh
                    dimensions = [h, w, l]
                    yaw = box3d.orientation.yaw_pitch_roll[0]

                    gt_world = np.array(
                        [[*box3d.center, *dimensions, yaw, 1.0]],
                        dtype=np.float32,
                    )

                    local_traj[track_id]["gt"].append(gt_world)

                    matched_pred, is_gt = self._match_gt_pred(
                        gt_world, box3d_class, preds
                    )
                    local_traj[track_id]["pred"].append(matched_pred)

                    if is_gt:
                        num_gt_boxes += 1
                    else:
                        num_pred_boxes += 1

                sample_token = sample["next"]

            for _, traj in local_traj.items():
                if len(traj["gt"]) >= self.min_seq_len:
                    trajectory = {
                        "gt": np.concatenate(traj["gt"]),
                        "pred": np.concatenate(traj["pred"]),
                    }
                    total_traj.append(trajectory)

        rank_zero_info(f"Use {num_gt_boxes} gt boxes.")
        rank_zero_info(f"Use {num_pred_boxes} pred boxes.")

        return total_traj

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> DictData:
        """Return the item at the given index.

        The trajectory will be randomly cropped to the minimum sequence length.
        """
        trajectory = self.samples[idx]
        data_dict: DictData = {}

        traj_len = len(trajectory["gt"])

        if traj_len > self.min_seq_len:
            first_frame = np.random.randint(traj_len - self.min_seq_len)
        else:
            first_frame = 0

        data_dict["gt_traj"] = trajectory["gt"][
            first_frame : first_frame + self.min_seq_len
        ]

        data_dict["pred_traj"] = trajectory["pred"][
            first_frame : first_frame + self.min_seq_len
        ]

        return data_dict
