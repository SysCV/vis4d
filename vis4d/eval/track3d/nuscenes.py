""""NuScenes evaluation code."""
from ..base import Evaluator
from vis4d.common import MetricLogs
import numpy as np

from vis4d.data.datasets.nuscenes import nuscenes_track_map
from nuscenes.utils.data_classes import Quaternion
from scipy.spatial.transform import Rotation as R

import json


class NuScenesEvaluator(Evaluator):

    inv_nuscenes_track_map = {v: k for k, v in nuscenes_track_map.items()}

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

    def __init__(self, split: str) -> None:
        super().__init__()
        self.split = split
        self.reset()

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return f"NuScenesEvaluator"

    @property
    def metrics(self) -> list[str]:
        return ["detect_3d", "track_3d"]

    def reset(self) -> None:
        self.tracks_3d = {}
        self.detect_3d = {}

    def get_attributes(
        self, name: str, velocity, velocity_thres: float = 1.0
    ) -> str:
        """Get nuScenes attributes."""
        if np.sqrt(velocity[0] ** 2 + velocity[1] ** 2) > velocity_thres:
            if name in [
                "car",
                "construction_vehicle",
                "bus",
                "truck",
                "trailer",
            ]:
                attr = "vehicle.moving"
            elif name in ["bicycle", "motorcycle"]:
                attr = "cycle.with_rider"
            else:
                attr = self.DefaultAttribute[name]
        else:
            if name in ["pedestrian"]:
                attr = "pedestrian.standing"
            elif name in ["bus"]:
                attr = "vehicle.stopped"
            else:
                attr = self.DefaultAttribute[name]
        return attr

    def _process_track_3d(self, data, outputs) -> None:
        annos = []
        token = data["CAM_FRONT"]["token"][0]
        if len(outputs.boxes_3d) != 0:
            for track_id, box_3d, score_3d, class_id in zip(
                outputs.track_ids,
                outputs.boxes_3d,
                outputs.scores_3d,
                outputs.class_ids,
            ):
                category = self.inv_nuscenes_track_map[
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

    def _process_detect_3d(self, data, outputs) -> None:
        annos = []
        token = data["CAM_FRONT"]["token"][0]
        if len(outputs.boxes_3d) != 0:
            for box_3d, score_3d, class_id in zip(
                outputs.boxes_3d,
                outputs.scores_3d,
                outputs.class_ids,
            ):
                category = self.inv_nuscenes_track_map[
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

                attribute_name = self.get_attributes(category, velocity)

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

    def process(self, data, outputs) -> None:
        self._process_detect_3d(data, outputs)
        self._process_track_3d(data, outputs)

    def evaluate(
        self,
        metric: str,
    ) -> tuple[MetricLogs, str]:
        # TODO: Add nuscenes eval code.
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
            result_file = f"./vis4d-workspace/nusc_{self.split}/track_3d_predictions.json"
        elif metric == "detect_3d":
            nusc_annos = {
                "results": self.detect_3d,
                "meta": metadata,
            }
            result_file = f"./vis4d-workspace/nusc_{self.split}/detect_3d_predictions.json"

        with open(result_file, mode="w", encoding="utf-8") as f:
            json.dump(nusc_annos, f)

        return {}, "Currently only save the json files."
