"""Load and convert NuScenes labels to scalabel format."""
import json
import os
import shutil
from typing import Dict, List, Tuple, Union

import numpy as np
from pytorch_lightning.utilities.distributed import rank_zero_warn
from scalabel.label.io import load, load_label_config, save
from scalabel.label.to_nuscenes import to_nuscenes
from scalabel.label.typing import Dataset, Frame

from vis4d.struct import ArgsType, MetricLogs

from .base import BaseDatasetLoader, _eval_mapping

try:  # pragma: no cover
    from nuscenes import NuScenes as nusc_data
    from nuscenes.eval.common.config import config_factory as track_configs
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.eval.tracking.evaluate import TrackingEval as track_eval
    from nuscenes.eval.tracking.utils import metric_name_to_print_format

    # pylint: disable=ungrouped-imports
    from scalabel.label.from_nuscenes import from_nuscenes

    NUSC_INSTALLED = True
except (ImportError, NameError):
    NUSC_INSTALLED = False


class NuScenes(BaseDatasetLoader):  # pragma: no cover
    """NuScenes dataloading class."""

    def __init__(
        self,
        version: str,
        split: str,
        add_non_key: bool,
        *args: ArgsType,
        tmp_dir: str = "./nuScenes_tmp/",
        metadata: Tuple[str, ...] = ("use_camera",),
        **kwargs: ArgsType,
    ):
        """Init dataset loader."""
        self.version = version
        self.split = split
        self.add_non_key = add_non_key
        self.tmp_dir = tmp_dir
        self.metadata = metadata
        super().__init__(*args, **kwargs)

    def load_dataset(self) -> Dataset:
        """Convert NuScenes annotations to Scalabel format."""
        assert (
            NUSC_INSTALLED
        ), "Using NuScenes dataset needs NuScenes devkit installed!."

        # annotations is the path to the label file in scalabel format.
        # if the file exists load it, else create it to that location
        assert (
            self.annotations is not None
        ), "Need a path to an annotation file to either load or create it."
        if not os.path.exists(self.annotations):
            dataset = from_nuscenes(
                self.data_root,
                self.version,
                self.split,
                self.num_processes,
                self.add_non_key,
            )
            save(self.annotations, dataset)
        else:
            # Load labels from existing file
            dataset = load(
                self.annotations,
                nprocs=self.num_processes,
            )

        if self.config_path is not None:
            dataset.config = load_label_config(self.config_path)

        return dataset

    def _convert_predictions(
        self,
        frames: List[Frame],
        mode: str,
    ) -> str:
        """Convert predictions back to nuScenes format, save out to tmp_dir."""
        os.makedirs(self.tmp_dir)

        metadata = {
            "use_camera": False,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        for m in self.metadata:
            metadata[m] = True

        result_path = os.path.join(self.tmp_dir, f"{mode}_results.json")

        nusc_results = to_nuscenes(Dataset(frames=frames), mode, metadata)

        with open(result_path, mode="w", encoding="utf-8") as f:
            json.dump(nusc_results, f)

        return result_path

    @staticmethod
    def _parse_detect_high_level_metrics(
        tp_errors: Dict[str, float],
        mean_ap: float,
        nd_score: float,
        eval_time: float,
    ) -> Tuple[List[str], Union[int, float], Union[int, float]]:
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
        str_summary_list: List[str],
        class_aps: Dict[str, float],
        class_tps: Dict[str, Dict[str, float]],
    ) -> List[str]:
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

    def _eval_detection(
        self,
        result_path: str,
        eval_set: str,
    ) -> Tuple[MetricLogs, str]:
        """Evaluate detection."""
        nusc = nusc_data(
            version=self.version,
            dataroot=self.data_root,
            verbose=False,
        )

        try:  # pragma: no cover
            nusc_eval = NuScenesEval(
                nusc,
                config=config_factory("detection_cvpr_2019"),
                result_path=result_path,
                eval_set=eval_set,
                output_dir=self.tmp_dir,
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

            log_dict = {
                "mAP": mean_ap,
                "NDS": nd_score,
            }
            str_summary = "\n".join(str_summary_list)

        except AssertionError as e:
            error_msg = "".join(e.args)
            rank_zero_warn(f"Evaluation error: {error_msg}")
            log_dict = {
                "mAP": 0,
                "NDS": 0,
            }
            str_summary = (
                "Evaluation failure might be raised due to sanity check"
            )
            rank_zero_warn(str_summary)

        # clean up tmp dir
        shutil.rmtree(self.tmp_dir)

        return log_dict, str_summary

    def _eval_tracking(
        self,
        result_path: str,
        eval_set: str,
    ) -> Tuple[MetricLogs, str]:
        """Evaluate tracking."""
        try:  # pragma: no cover
            nusc_eval = track_eval(
                config=track_configs("tracking_nips_2019"),
                result_path=result_path,
                eval_set=eval_set,
                output_dir=self.tmp_dir,
                verbose=False,
                nusc_version=self.version,
                nusc_dataroot=self.data_root,
            )
            metrics, _ = nusc_eval.evaluate()

            str_summary_list = ["\nPer-class results:"]
            metric_names = metrics.label_metrics.keys()
            str_summary_list.append(
                "\t\t" + "\t".join([m.upper() for m in metric_names])
            )

            class_names = metrics.class_names
            max_name_length = 7
            for class_name in class_names:
                print_class = class_name[:max_name_length].ljust(
                    max_name_length + 1
                )

                for metric_name in metric_names:
                    val = metrics.label_metrics[metric_name][class_name]
                    print_format = (
                        "%f"
                        if np.isnan(val)
                        else metric_name_to_print_format(metric_name)
                    )
                    print_class += f"\t{(print_format % val)}"

                str_summary_list.append(print_class)

            str_summary_list.append("\nAggregated results:")
            for metric_name in metric_names:
                val = metrics.compute_metric(metric_name, "all")
                print_format = metric_name_to_print_format(metric_name)
                str_summary_list.append(
                    f"{metric_name.upper()}\t{print_format % val}"
                )
            str_summary_list.append(f"Eval time: {metrics.eval_time:.1f}s")

            log_dict = {
                "AMOTA": metrics.compute_metric("amota", "all"),
                "AMOTP": metrics.compute_metric("amotp", "all"),
            }
            str_summary = "\n".join(str_summary_list)
        except AssertionError as e:
            error_msg = "".join(e.args)
            rank_zero_warn(f"Evaluation error: {error_msg}")
            log_dict = {
                "aMOTA": 0,
                "MOTP": 0,
            }
            str_summary = (
                "Evaluation failure might be raised due to sanity check"
                + " or motmetrics version is not 1.13.0"
                + " or numpy version is not <= 1.19"
            )
            rank_zero_warn(str_summary)

        # clean up tmp dir
        shutil.rmtree(self.tmp_dir)

        return log_dict, str_summary

    def evaluate(
        self, metric: str, predictions: List[Frame], gts: List[Frame]
    ) -> Tuple[MetricLogs, str]:
        """Evaluate according to nuScenes metrics."""
        if not metric in ["detect_3d", "track_3d"]:
            return super().evaluate(metric, predictions, gts)

        if metric == "detect_3d":
            mode = "detection"
        else:
            mode = "tracking"

        result_path = self._convert_predictions(predictions, mode)

        if "mini" in self.version:
            eval_set = "mini_val"
        else:
            eval_set = "val"

        if mode == "detection":
            log_dict, str_summary = self._eval_detection(result_path, eval_set)
        else:
            log_dict, str_summary = self._eval_tracking(result_path, eval_set)

        return log_dict, str_summary

    def _check_metrics(self) -> None:
        """Check if evaluation metrics specified are valid."""
        assert self.eval_metrics is not None
        for metric in self.eval_metrics:
            if (
                metric not in ["detect_3d", "track_3d"]
                and metric not in _eval_mapping
            ):  # pragma: no cover
                raise KeyError(
                    f"metric {metric} is not supported in {self.name}"
                )
