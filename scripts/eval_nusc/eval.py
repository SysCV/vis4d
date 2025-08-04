"""nuScenes evaluation pipeline for Vis4D."""

import argparse
import os
import json

from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory

from nuscenes.eval.tracking.evaluate import TrackingEval as track_eval
from nuscenes.eval.tracking.utils import print_final_metrics
from nuscenes.eval.tracking.data_classes import TrackingConfig, TrackingMetrics
from nuscenes.eval.common.config import config_factory as track_configs


def eval_detection(
    version: str,
    dataroot: str,
    output_dir: str,
    result_path: str,
    eval_set: str,
) -> None:
    """Evaluate detection results."""
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    nusc_eval = NuScenesEval(
        nusc,
        config=config_factory("detection_cvpr_2019"),
        result_path=result_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
    )
    _ = nusc_eval.main(render_curves=False)


def eval_tracking(
    version: str, output_dir: str, result_path: str, root: str, eval_set: str
) -> None:
    """Evaluate tracking results."""
    nusc_eval = track_eval(
        config=track_configs("tracking_nips_2019"),
        result_path=result_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
        nusc_version=version,
        nusc_dataroot=root,
    )
    _ = nusc_eval.main()


def evaluate(
    version: str,
    dataroot: str,
    mode: str,
    output_dir: str,
    result_path: str,
    root: str,
) -> None:
    """nuScenes evaluation."""
    if "mini" in version:
        eval_set = "mini_val"
    else:
        eval_set = "val"

    if mode == "tracking":
        eval_tracking(version, output_dir, result_path, root, eval_set)
    else:
        eval_detection(version, dataroot, output_dir, result_path, eval_set)


def print_metric_summary(metric_summary_path: str) -> None:
    """Print metric summary."""
    with open(metric_summary_path, "r") as f:
        metrics = json.load(f)

    cfg = TrackingConfig.deserialize(metrics["cfg"])
    tm = TrackingMetrics(cfg=cfg)
    tm.add_runtime(metrics["eval_time"])
    tm.label_metrics = metrics["label_metrics"]

    print_final_metrics(metrics)


if __name__ == "__main__":
    """Main."""
    parser = argparse.ArgumentParser(description="NuScenes eval for Vis4D.")
    parser.add_argument(
        "--input",
        "-i",
        help=(
            "Folder path to the nuScenes format detection / tracking results."
        ),
    )
    parser.add_argument(
        "--version",
        "-v",
        choices=["v1.0-trainval", "v1.0-test", "v1.0-mini"],
        help="NuScenes dataset version to convert.",
    )
    parser.add_argument(
        "--dataroot",
        "-d",
        help="NuScenes dataset root.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="tracking",
        choices=["tracking", "detection"],
        help="Conversion mode: detection or tracking.",
    )
    args = parser.parse_args()

    if args.mode == "detection":
        metric = "detect_3d"
    else:
        metric = "track_3d"

    evaluate(
        args.version,
        args.dataroot,
        args.mode,
        args.input,
        os.path.join(args.input, f"{metric}_predictions.json"),
        args.dataroot,
    )
