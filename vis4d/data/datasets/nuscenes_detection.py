"""NuScenes multi-sensor video dataset."""
from __future__ import annotations

import json

import numpy as np

from vis4d.common.typing import ArgsType, DictStrAny, NDArrayF32, NDArrayI64
from vis4d.data.typing import DictData
from .nuscenes import NuScenes, nuscenes_class_map, nusc_tracking_cats


class NuScenesDetection(NuScenes):
    """NuScenes detection dataset."""

    def __init__(self, detection_result: str, **kwargs: ArgsType) -> None:
        """Creates an instance of the class."""
        self.detection_result = detection_result

        with open(self.detection_result) as f:
            self.predictions = json.load(f)

        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """Concise representation of the dataset."""
        return (
            f"NuScenesDetection {self.version} {self.split} using "
            + f"{self.detection_result}"
        )

    def _load_pred(
        self, preds: list[DictStrAny]
    ) -> tuple[NDArrayF32, NDArrayI64, NDArrayF32, NDArrayF32]:
        """Load nuscenes format prediction."""
        boxes3d = np.empty((1, 10), dtype=np.float32)[1:]
        boxes3d_classes = np.empty((1,), dtype=np.int64)[1:]
        boxes3d_scores = np.empty((1,), dtype=np.float32)[1:]
        boxes3d_velocities = np.empty((1, 3), dtype=np.float32)[1:]

        for pred in preds:
            if pred["detection_name"] not in nuscenes_class_map:
                continue

            boxes3d = np.concatenate(
                [
                    boxes3d,
                    np.array(
                        [
                            [
                                *pred["translation"],
                                *pred["size"],
                                *pred["rotation"],
                            ]
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
            boxes3d_classes = np.concatenate(
                [
                    boxes3d_classes,
                    np.array(
                        [nuscenes_class_map[pred["detection_name"]]],
                        dtype=np.int64,
                    ),
                ]
            )
            boxes3d_scores = np.concatenate(
                [
                    boxes3d_scores,
                    np.array([pred["detection_score"]], dtype=np.float32),
                ]
            )
            boxes3d_velocities = np.concatenate(
                [
                    boxes3d_velocities,
                    np.array([[*pred["velocity"], 0]], dtype=np.float32),
                ]
            )

        return boxes3d, boxes3d_classes, boxes3d_scores, boxes3d_velocities

    def __getitem__(self, idx: int) -> DictData:
        """Get single sample.

        Args:
            idx (int): Index of sample.

        Returns:
            DictData: sample at index in Vis4D input format.
        """
        data_dict = super().__getitem__(idx)

        (
            data_dict["pred_boxes3d"],
            data_dict["pred_boxes3d_classes"],
            data_dict["pred_boxes3d_scores"],
            data_dict["pred_boxes3d_velocities"],
        ) = self._load_pred(self.predictions["results"][data_dict["token"]])

        return data_dict
