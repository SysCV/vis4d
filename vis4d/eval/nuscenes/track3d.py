"""NuScenes 3D tracking evaluation code."""
from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from typing import Any

from nuscenes.utils.data_classes import Quaternion

from vis4d.common.array import array_to_numpy
from vis4d.common.typing import ArrayLike, DictStrAny, MetricLogs
from vis4d.data.datasets.nuscenes import nuscenes_class_map

from ..base import Evaluator


class NuScenesTrack3DEvaluator(Evaluator):
    """NuScenes 3D tracking evaluation class."""

    inv_nuscenes_class_map = {v: k for k, v in nuscenes_class_map.items()}

    tracking_cats = [
        "bicycle",
        "motorcycle",
        "pedestrian",
        "bus",
        "car",
        "trailer",
        "truck",
    ]

    def __init__(self, metadata: tuple[str, ...] = ("use_camera",)) -> None:
        """Initialize NuScenes evaluator."""
        super().__init__()
        self.meta_data = {
            "use_camera": False,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        for m in metadata:
            self.meta_data[m] = True

        self.tracks_3d: DictStrAny = {}
        self.reset()

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return "NuScenes 3D Tracking Evaluator"

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return ["track_3d"]

    def gather(  # type: ignore
        self, gather_func: Callable[[Any], Any]
    ) -> None:
        """Gather variables in case of distributed setting (if needed).

        Args:
            gather_func (Callable[[Any], Any]): Gather function.
        """
        tracks_3d_list = gather_func(self.tracks_3d)
        if tracks_3d_list is not None:
            prediction_list = [p.items() for p in tracks_3d_list]
            self.tracks_3d = dict(itertools.chain(*prediction_list))

    def reset(self) -> None:
        """Reset evaluator."""
        self.tracks_3d.clear()

    def _process_track_3d(
        self,
        token: str,
        boxes_3d: ArrayLike,
        velocities: ArrayLike,
        scores_3d: ArrayLike,
        class_ids: ArrayLike,
        track_ids: ArrayLike,
    ) -> None:
        """Process 3D tracking results."""
        annos = []
        if len(boxes_3d) != 0:
            for box_3d, velocity, score_3d, class_id, track_id in zip(
                boxes_3d,
                velocities,
                scores_3d,
                class_ids,
                track_ids,
            ):
                box_3d = array_to_numpy(box_3d)
                velocity = array_to_numpy(velocity)
                score_3d = array_to_numpy(score_3d)
                class_id = array_to_numpy(class_id)
                track_id = array_to_numpy(track_id)

                category = self.inv_nuscenes_class_map[int(class_id)]
                if not category in self.tracking_cats:
                    continue

                translation = box_3d[0:3]

                dimension = box_3d[3:6]

                rotation = Quaternion(box_3d[6:].tolist())

                score = float(score_3d)

                velocity_list = velocity.tolist()

                nusc_anno = {
                    "sample_token": token,
                    "translation": translation.tolist(),
                    "size": dimension.tolist(),
                    "rotation": rotation.elements.tolist(),
                    "velocity": [velocity_list[0], velocity_list[1]],
                    "tracking_id": int(track_id),
                    "tracking_name": category,
                    "tracking_score": score,
                }
                annos.append(nusc_anno)
        self.tracks_3d[token] = annos

    def process_batch(  # type: ignore # pylint: disable=arguments-differ
        self,
        tokens: list[str] | str,
        boxes_3d: ArrayLike,
        velocities: ArrayLike,
        class_ids: ArrayLike,
        scores_3d: ArrayLike,
        track_ids: ArrayLike,
    ) -> None:
        """Process the results."""
        # Currently only support batch size of 1.
        if isinstance(tokens, list):
            tokens = sum(tokens, [])
            token = tokens[0]
            assert all(
                token == t for t in tokens
            ), "Tokens should be the same."
        else:
            token = tokens

        self._process_track_3d(
            token, boxes_3d, velocities, scores_3d, class_ids, track_ids
        )

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate the results."""
        return {}, "Currently only save the json files."

    def save(self, metric: str, output_dir: str) -> None:
        """Save the results to json files."""
        nusc_annos = {
            "results": self.tracks_3d,
            "meta": self.meta_data,
        }
        result_file = f"{output_dir}/track_3d_predictions.json"

        with open(result_file, mode="w", encoding="utf-8") as f:
            json.dump(nusc_annos, f)
