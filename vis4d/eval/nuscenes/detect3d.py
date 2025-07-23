"""NuScenes 3D detection evaluation code."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

import numpy as np

from vis4d.common.array import array_to_numpy
from vis4d.common.imports import NUSCENES_AVAILABLE
from vis4d.common.logging import rank_zero_warn
from vis4d.common.typing import ArrayLike, DictStrAny, MetricLogs
from vis4d.data.datasets.nuscenes import (
    nuscenes_attribute_map,
    nuscenes_class_map,
)

from ..base import Evaluator

if NUSCENES_AVAILABLE:
    from nuscenes import NuScenes as NuScenesDevkit
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.utils.data_classes import Quaternion
else:
    raise ImportError("nuscenes-devkit is not installed.")


def _parse_high_level_metrics(
    mean_ap: float,
    tp_errors: dict[str, float],
    nd_score: float,
    eval_time: float,
) -> tuple[MetricLogs, list[str]]:
    """Collect high-level metrics."""
    log_dict: MetricLogs = {
        "mAP": mean_ap,
        "mATE": tp_errors["trans_err"],
        "mASE": tp_errors["scale_err"],
        "mAOE": tp_errors["orient_err"],
        "mAVE": tp_errors["vel_err"],
        "mAAE": tp_errors["attr_err"],
        "NDS": nd_score,
    }

    str_summary_list = ["\nHigh-level metrics:"]
    for k, v in log_dict.items():
        str_summary_list.append(f"{k}: {v:.4f}")

    str_summary_list.append(f"Eval time: {eval_time:.1f}s")

    return log_dict, str_summary_list


def _parse_per_class_metrics(
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


class NuScenesDet3DEvaluator(Evaluator):
    """NuScenes 3D detection evaluation class."""

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

    def __init__(
        self,
        data_root: str,
        version: str,
        split: str,
        save_only: bool = False,
        class_map: dict[str, int] | None = None,
        metadata: tuple[str, ...] = ("use_camera",),
        use_default_attr: bool = False,
        velocity_thres: float = 1.0,
    ) -> None:
        """Initialize NuScenes evaluator."""
        super().__init__()
        self.data_root = data_root
        self.version = version
        self.split = split
        self.save_only = save_only
        self.use_default_attr = use_default_attr
        self.velocity_thres = velocity_thres

        self.meta_data = {
            "use_camera": False,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        for m in metadata:
            self.meta_data[m] = True

        class_map = class_map or nuscenes_class_map
        self.inv_nuscenes_class_map = {v: k for k, v in class_map.items()}

        self.output_dir = ""
        self.detect_3d: DictStrAny = {}
        self.reset()

    def __repr__(self) -> str:
        """Concise representation of the dataset evaluator."""
        return "NuScenes 3D Detection Evaluator"

    @property
    def metrics(self) -> list[str]:
        """Supported metrics."""
        return ["detect_3d"]

    def gather(  # type: ignore
        self, gather_func: Callable[[Any], Any]
    ) -> None:
        """Gather variables in case of distributed setting (if needed).

        Args:
            gather_func (Callable[[Any], Any]): Gather function.
        """
        detect_3d_list = gather_func(self.detect_3d)
        if detect_3d_list is not None:
            collated_detect_3d: DictStrAny = {}
            for prediction in detect_3d_list:
                for k, v in prediction.items():
                    if k not in collated_detect_3d:
                        collated_detect_3d[k] = v
                    else:
                        collated_detect_3d[k].extend(v)
            self.detect_3d = collated_detect_3d

    def reset(self) -> None:
        """Reset evaluator."""
        self.detect_3d.clear()

    def get_attributes(self, name: str, velocity: list[float]) -> str:
        """Get nuScenes attributes."""
        if self.use_default_attr:
            return self.DefaultAttribute[name]

        if np.sqrt(velocity[0] ** 2 + velocity[1] ** 2) > self.velocity_thres:
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

    def _process_detect_3d(
        self,
        token: str,
        boxes_3d: ArrayLike,
        velocities: ArrayLike,
        scores_3d: ArrayLike,
        class_ids: ArrayLike,
        attributes: ArrayLike | None = None,
    ) -> None:
        """Process 3D detection results."""
        annos = []
        boxes_3d_np = array_to_numpy(boxes_3d, n_dims=None, dtype=np.float32)
        velocities_np = array_to_numpy(
            velocities, n_dims=None, dtype=np.float32
        )
        scores_3d_np = array_to_numpy(scores_3d, n_dims=None, dtype=np.float32)
        class_ids_np = array_to_numpy(class_ids, n_dims=None, dtype=np.int64)

        if len(boxes_3d_np) != 0:
            for i, (box_3d, velocity, score_3d, class_id) in enumerate(
                zip(
                    boxes_3d_np,
                    velocities_np,
                    scores_3d_np,
                    class_ids_np,
                )
            ):
                category = self.inv_nuscenes_class_map[int(class_id)]

                translation = box_3d[0:3]

                dims = box_3d[3:6].tolist()
                dimension = [d if d >= 0 else 0.1 for d in dims]

                rotation = Quaternion(box_3d[6:].tolist())

                score = float(score_3d)

                velocity_list = velocity.tolist()

                if attributes is None:
                    attribute_name = self.get_attributes(
                        category, velocity_list
                    )
                else:
                    attribute = array_to_numpy(
                        attributes[i], n_dims=None, dtype=np.int64  # type: ignore # pylint: disable=line-too-long
                    )
                    attribute_name = self.inv_nuscenes_attribute_map[
                        int(attribute)
                    ]

                nusc_anno = {
                    "sample_token": token,
                    "translation": translation.tolist(),
                    "size": dimension,
                    "rotation": rotation.elements.tolist(),
                    "velocity": [velocity_list[0], velocity_list[1]],
                    "detection_name": category,
                    "detection_score": score,
                    "attribute_name": attribute_name,
                }
                annos.append(nusc_anno)
        self.detect_3d[token] = annos

    def process_batch(
        self,
        tokens: list[str],
        boxes_3d: list[ArrayLike],
        velocities: list[ArrayLike],
        class_ids: list[ArrayLike],
        scores_3d: list[ArrayLike],
        attributes: list[ArrayLike] | None = None,
    ) -> None:
        """Process the results."""
        for i, token in enumerate(tokens):
            self._process_detect_3d(
                token,
                boxes_3d[i],
                velocities[i],
                scores_3d[i],
                class_ids[i],
                attributes[i] if attributes is not None else None,
            )

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate the results."""
        assert metric == "detect_3d"
        if self.save_only:
            return {}, "Results are saved to the json file."

        try:
            nusc = NuScenesDevkit(
                version=self.version,
                dataroot=self.data_root,
                verbose=False,
            )

            nusc_eval = NuScenesEval(
                nusc,
                config=config_factory("detection_cvpr_2019"),
                result_path=f"{self.output_dir}/detect_3d_predictions.json",
                eval_set=self.split,
                output_dir=os.path.join(self.output_dir, "detection"),
            )
            metrics, _ = nusc_eval.evaluate()
            metrics_summary = metrics.serialize()

            log_dict, str_summary_list = _parse_high_level_metrics(
                metrics_summary["mean_ap"],
                metrics_summary["tp_errors"],
                metrics_summary["nd_score"],
                metrics_summary["eval_time"],
            )

            class_aps = metrics_summary["mean_dist_aps"]
            class_tps = metrics_summary["label_tp_errors"]
            str_summary_list = _parse_per_class_metrics(
                str_summary_list, class_aps, class_tps
            )

            str_summary = "\n".join(str_summary_list)
        except Exception as e:  # pylint: disable=broad-except
            error_msg = "".join(e.args)
            rank_zero_warn(f"Evaluation error: {error_msg}")
            log_dict = {}
            str_summary = (
                "Evaluation failure might be raised due to sanity check"
                + "or all emtpy boxes."
            )
            rank_zero_warn(str_summary)
        return log_dict, str_summary

    def save(self, metric: str, output_dir: str) -> None:
        """Save the results to json files."""
        assert metric == "detect_3d"
        nusc_annos = {"results": self.detect_3d, "meta": self.meta_data}
        result_file = f"{output_dir}/detect_3d_predictions.json"

        with open(result_file, mode="w", encoding="utf-8") as f:
            json.dump(nusc_annos, f)

        self.output_dir = output_dir
