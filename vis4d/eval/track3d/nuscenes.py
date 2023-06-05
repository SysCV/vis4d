"""NuScenes evaluation code."""
import itertools
import json
from collections.abc import Callable
from typing import Any

import numpy as np
from nuscenes.utils.data_classes import Quaternion
from scipy.spatial.transform import Rotation as R
from torch import Tensor

from vis4d.common import DictStrAny, MetricLogs
from vis4d.data.datasets.nuscenes import (
    nuscenes_attribute_map,
    nuscenes_class_map,
)

from ..base import Evaluator


# TODO: Refactor it to work with our own boxes3d
class NuScenesEvaluator(Evaluator):
    """NuScenes 3D detection and tracking evaluation class."""

    inv_nuscenes_class_map = {v: k for k, v in nuscenes_class_map.items()}
    inv_nuscenes_attribute_map = {
        v: k for k, v in nuscenes_attribute_map.items()
    }

    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }

    tracking_cats = [
        "bicycle",
        "motorcycle",
        "pedestrian",
        "bus",
        "car",
        "trailer",
        "truck",
    ]

    def __init__(self) -> None:
        """Initialize NuScenes evaluator."""
        super().__init__()
        self.detect_3d: DictStrAny = {}
        self.tracks_3d: DictStrAny = {}
        self.reset()

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return "NuScenesEvaluator"

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return ["detect_3d", "track_3d"]

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

        detect_3d_list = gather_func(self.detect_3d)
        if detect_3d_list is not None:
            prediction_list = [p.items() for p in detect_3d_list]
            self.detect_3d = dict(itertools.chain(*prediction_list))

    def reset(self) -> None:
        """Reset evaluator."""
        self.tracks_3d.clear()
        self.detect_3d.clear()

    def get_attributes(
        self, name: str, velocity: list[float], velocity_thres: float = 1.0
    ) -> str:
        """Get nuScenes attributes."""
        if np.sqrt(velocity[0] ** 2 + velocity[1] ** 2) > velocity_thres:
            if name in {
                "car",
                "construction_vehicle",
                "bus",
                "truck",
                "trailer",
            }:
                attr = "vehicle.moving"
            elif name in {"bicycle", "motorcycle"}:
                attr = "cycle.with_rider"
            else:
                attr = self.DefaultAttribute[name]
        elif name in {"pedestrian"}:
            attr = "pedestrian.standing"
        elif name in {"bus"}:
            attr = "vehicle.stopped"
        else:
            attr = self.DefaultAttribute[name]
        return attr

    def _process_track_3d(
        self,
        token: str,
        boxes_3d: Tensor,
        scores_3d: Tensor,
        class_ids: Tensor,
        track_ids: Tensor,
    ) -> None:
        """Process 3D tracking results."""
        annos = []
        if len(boxes_3d) != 0:
            for track_id, box_3d, score_3d, class_id in zip(
                track_ids,
                boxes_3d,
                scores_3d,
                class_ids,
            ):
                category = self.inv_nuscenes_class_map[
                    int(class_id.cpu().numpy())
                ]
                if not category in self.tracking_cats:
                    continue

                translation = box_3d[0:3].cpu().numpy()

                dim = box_3d[3:6].cpu().numpy().tolist()
                dimension = [dim[1], dim[2], dim[0]]

                # Using extrinsic rotation here to align with Pytorch3D
                x, y, z, w = R.from_euler(
                    "XYZ", box_3d[6:9].cpu().numpy()
                ).as_quat()
                rotation = Quaternion([w, x, y, z])

                score = float(score_3d.cpu().numpy())

                velocity = box_3d[9:12].cpu().numpy().tolist()

                nusc_anno = {
                    "sample_token": token,
                    "translation": translation.tolist(),
                    "size": dimension,
                    "rotation": rotation.elements.tolist(),
                    "velocity": [velocity[0], velocity[1]],
                    "tracking_id": int(track_id.cpu().numpy()),
                    "tracking_name": category,
                    "tracking_score": score,
                }
                annos.append(nusc_anno)
        self.tracks_3d[token] = annos

    def _process_detect_3d(
        self,
        token: str,
        boxes_3d: Tensor,
        scores_3d: Tensor,
        class_ids: Tensor,
        attributes: Tensor | None = None,
    ) -> None:
        """Process 3D detection results."""
        annos = []
        if len(boxes_3d) != 0:
            for i, (box_3d, score_3d, class_id) in enumerate(
                zip(
                    boxes_3d,
                    scores_3d,
                    class_ids,
                )
            ):
                category = self.inv_nuscenes_class_map[
                    int(class_id.cpu().numpy())
                ]

                translation = box_3d[0:3].cpu().numpy()

                dim = box_3d[3:6].cpu().numpy().tolist()
                dimension = [dim[1], dim[2], dim[0]]
                dimension = [d if d >= 0 else 0.1 for d in dimension]

                # Using extrinsic rotation here to align with Pytorch3D
                x, y, z, w = R.from_euler(
                    "XYZ", box_3d[6:9].cpu().numpy()
                ).as_quat()
                rotation = Quaternion([w, x, y, z])

                score = float(score_3d.cpu().numpy())

                velocity = box_3d[9:12].cpu().numpy().tolist()

                if attributes is None:
                    attribute_name = self.get_attributes(category, velocity)
                else:
                    attribute_name = self.inv_nuscenes_attribute_map[
                        int(attributes[i].cpu().numpy())
                    ]

                nusc_anno = {
                    "sample_token": token,
                    "translation": translation.tolist(),
                    "size": dimension,
                    "rotation": rotation.elements.tolist(),
                    "velocity": [velocity[0], velocity[1]],
                    "detection_name": category,
                    "detection_score": score,
                    "attribute_name": attribute_name,
                }
                annos.append(nusc_anno)
        self.detect_3d[token] = annos

    def process(  # type: ignore # pylint: disable=arguments-differ
        self,
        tokens: list[str] | str,
        boxes_3d: Tensor,
        scores_3d: Tensor,
        class_ids: Tensor,
        track_ids: Tensor,
        attributes: Tensor | None = None,
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

        self._process_detect_3d(
            token,
            boxes_3d,
            scores_3d,
            class_ids,
            attributes,
        )
        self._process_track_3d(
            token, boxes_3d, scores_3d, class_ids, track_ids
        )

    def evaluate(
        self,
        metric: str,
    ) -> tuple[MetricLogs, str]:
        """Evaluate the results."""
        # TODO: Add nuscenes eval code.
        return {}, "Currently only save the json files."

    def save(self, metric: str, output_dir: str) -> None:
        """Save the results to json files."""
        metadata = {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
        if metric == "track_3d":
            nusc_annos = {
                "results": self.tracks_3d,
                "meta": metadata,
            }
            result_file = f"{output_dir}/track_3d_predictions.json"
        elif metric == "detect_3d":
            nusc_annos = {
                "results": self.detect_3d,
                "meta": metadata,
            }
            result_file = f"{output_dir}/detect_3d_predictions.json"

        with open(result_file, mode="w", encoding="utf-8") as f:
            json.dump(nusc_annos, f)
