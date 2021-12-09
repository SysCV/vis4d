"""Load and convert NuScenes labels to scalabel format."""
import json
import os
import shutil
from typing import Dict, List, Tuple, Union

from scalabel.label.io import load, load_label_config, save
from scalabel.label.to_nuscenes import to_nuscenes
from scalabel.label.typing import Dataset, Frame

from vis4d.struct import MetricLogs

from .base import BaseDatasetConfig, BaseDatasetLoader

try:  # pragma: no cover
    from nuscenes import NuScenes as nusc_data
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    # pylint: disable=ungrouped-imports
    from scalabel.label.from_nuscenes import from_nuscenes

    NUSC_INSTALLED = True
except (ImportError, NameError):
    NUSC_INSTALLED = False


class NuScenesDatasetConfig(BaseDatasetConfig):
    """Config for training/evaluation datasets."""

    version: str
    split: str
    add_non_key: bool
    tmp_dir: str = "./nuScenes_tmp/"
    metadata: List[str] = ["use_camera"]


class NuScenes(BaseDatasetLoader):  # pragma: no cover
    """NuScenes dataloading class."""

    def __init__(self, cfg: BaseDatasetConfig):
        """Init dataset loader."""
        super().__init__(cfg)
        self.cfg: NuScenesDatasetConfig = NuScenesDatasetConfig(**cfg.dict())

    def load_dataset(self) -> Dataset:
        """Convert NuScenes annotations to Scalabel format."""
        assert (
            NUSC_INSTALLED
        ), "Using NuScenes dataset needs NuScenes devkit installed!."

        # cfg.annotations is the path to the label file in scalabel format.
        # if the file exists load it, else create it to that location
        assert (
            self.cfg.annotations is not None
        ), "Need a path to an annotation file to either load or create it."
        if not os.path.exists(self.cfg.annotations):
            dataset = from_nuscenes(
                self.cfg.data_root,
                self.cfg.version,
                self.cfg.split,
                self.cfg.num_processes,
                self.cfg.add_non_key,
            )
            save(self.cfg.annotations, dataset)
        else:
            # Load labels from existing file
            dataset = load(
                self.cfg.annotations,
                validate_frames=self.cfg.validate_frames,
                nprocs=self.cfg.num_processes,
            )

        if self.cfg.config_path is not None:
            dataset.config = load_label_config(self.cfg.config_path)

        return dataset

    def _convert_predictions(
        self,
        frames: List[Frame],
        mode: str,
    ) -> str:
        """Convert predictions back to nuScenes format, save out to tmp_dir."""
        os.makedirs(self.cfg.tmp_dir)

        metadata = {
            "use_camera": False,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        for m in self.cfg.metadata:
            metadata[m] = True

        result_path = os.path.join(self.cfg.tmp_dir, f"{mode}_results.json")

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
        str_summary_list = ["High-level metrics:"]
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
        str_summary_list.append("")
        str_summary_list.append("Per-class results:")
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
            version=self.cfg.version,
            dataroot=self.cfg.data_root,
            verbose=False,
        )
        nusc_eval = NuScenesEval(
            nusc,
            config=config_factory("detection_cvpr_2019"),
            result_path=result_path,
            eval_set=eval_set,
            output_dir=self.cfg.tmp_dir,
            verbose=False,
        )

        try:  # pragma: no cover
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

        except AssertionError:
            log_dict = {
                "mAP": 0,
                "NDS": 0,
            }
            str_summary = "Fail to evaluate due to sanity check or errors!"

        # clean up tmp dir
        shutil.rmtree(self.cfg.tmp_dir)

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

        if "mini" in self.cfg.version:
            eval_set = "mini_val"
        else:
            eval_set = "val"

        if mode == "detection":
            log_dict, str_summary = self._eval_detection(result_path, eval_set)

        return log_dict, str_summary

    def _check_metrics(self) -> None:
        """Check if evaluation metrics specified are valid."""
        for metric in self.cfg.eval_metrics:
            if metric not in ["detect_3d"]:  # pragma: no cover
                raise KeyError(
                    f"metric {metric} is not supported in {self.cfg.name}"
                )
