"""NuScenes evaluation code."""
from __future__ import annotations

import os
import itertools
import json
from collections.abc import Callable
from typing import Any

import numpy as np
from nuscenes.utils.data_classes import Quaternion
from scipy.spatial.transform import Rotation as R
from torch import Tensor

from vis4d.common.imports import NUSCENES_AVAILABLE
from vis4d.common.typing import DictStrAny, MetricLogs
from vis4d.data.datasets.nuscenes import (
    nuscenes_attribute_map,
    nuscenes_class_map,
)

from ..base import Evaluator

if NUSCENES_AVAILABLE:
    from nuscenes import NuScenes as NuScenesDevkit
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval


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

    def __init__(
        self,
        data_root: str,
        version: str,
        split: str,
        output_dir: str,
        metadata: tuple[str, ...] = ("use_camera",),
    ) -> None:
        """Initialize NuScenes evaluator."""
        super().__init__()
        self.data_root = data_root
        self.version = version
        self.split = split
        self.output_dir = output_dir

        self.meta_data = {
            "use_camera": False,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        for m in metadata:
            self.meta_data[m] = True

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

    def process_batch(  # type: ignore # pylint: disable=arguments-differ
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

    @staticmethod
    def _parse_detect_high_level_metrics(
        tp_errors: dict[str, float],
        mean_ap: float,
        nd_score: float,
        eval_time: float,
    ) -> tuple[list[str], int | float, int | float]:
        """Collect high-level metrics."""
        str_summary_list = ["\nHigh-level metrics:"]
        str_summary_list.append(f"mAP: {mean_ap:.4f}")
        err_name_mapping = {
            "trans_err": "mATE",
            "scale_err": "mASE",
            "orient_err": "mAOE",
            "vel_err": "mAVE",
            "attr_err": "mAAE",
        }
        for tp_name, tp_val in tp_errors.items():
            str_summary_list.append(
                f"{err_name_mapping[tp_name]}: {tp_val:.4f}"
            )
        str_summary_list.append(f"NDS: {nd_score:.4f}")
        str_summary_list.append(f"Eval time: {eval_time:.1f}s")

        if mean_ap == 0:
            mean_ap = int(mean_ap)
        if nd_score == 0:
            nd_score = int(nd_score)

        return str_summary_list, mean_ap, nd_score

    @staticmethod
    def _parse_detect_per_class_metrics(
        str_summary_list: list[str],
        class_aps: dict[str, float],
        class_tps: dict[str, dict[str, float]],
    ) -> list[str]:
        """Collect per-class metrics."""
        str_summary_list.append("\nPer-class results:")
        str_summary_list.append("Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE")

        for class_name in class_aps.keys():
            tmp_str_list = [class_name]
            tmp_str_list.append(f"{class_aps[class_name]:.3f}")
            tmp_str_list.append(f"{class_tps[class_name]['trans_err']:.3f}")
            tmp_str_list.append(f"{class_tps[class_name]['scale_err']:.3f}")
            tmp_str_list.append(f"{class_tps[class_name]['orient_err']:.3f}")
            tmp_str_list.append(f"{class_tps[class_name]['vel_err']:.3f}")
            tmp_str_list.append(f"{class_tps[class_name]['attr_err']:.3f}")

            str_summary_list.append("\t".join(tmp_str_list))
        return str_summary_list

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate the results."""
        output_dir = os.path.join(self.output_dir, metric)
        if metric == "detect_3d":
            nusc = NuScenesDevkit(
                version=self.version, dataroot=self.data_root, verbose=False
            )

            nusc_eval = NuScenesEval(
                nusc,
                config=config_factory("detection_cvpr_2019"),
                result_path=f"{output_dir}/detect_3d_predictions.json",
                eval_set=self.split,
                output_dir=os.path.join(output_dir, "detection"),
                verbose=False,
            )
            metrics, _ = nusc_eval.evaluate()
            metrics_summary = metrics.serialize()

            (
                str_summary_list,
                mean_ap,
                nd_score,
            ) = self._parse_detect_high_level_metrics(
                metrics_summary["tp_errors"],
                metrics_summary["mean_ap"],
                metrics_summary["nd_score"],
                metrics_summary["eval_time"],
            )

            class_aps = metrics_summary["mean_dist_aps"]
            class_tps = metrics_summary["label_tp_errors"]
            str_summary_list = self._parse_detect_per_class_metrics(
                str_summary_list, class_aps, class_tps
            )

            log_dict = {"mAP": mean_ap, "NDS": nd_score}
            str_summary = "\n".join(str_summary_list)
            return log_dict, str_summary
        else:
            return {}, "Currently only save the json files."

    def save(self, metric: str, output_dir: str) -> None:
        """Save the results to json files."""
        if metric == "track_3d":
            nusc_annos = {
                "results": self.tracks_3d,
                "meta": self.meta_data,
            }
            result_file = f"{output_dir}/track_3d_predictions.json"
        elif metric == "detect_3d":
            nusc_annos = {
                "results": self.detect_3d,
                "meta": self.meta_data,
            }
            result_file = f"{output_dir}/detect_3d_predictions.json"

        with open(result_file, mode="w", encoding="utf-8") as f:
            json.dump(nusc_annos, f)
