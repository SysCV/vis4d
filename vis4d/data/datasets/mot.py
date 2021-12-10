"""Load MOTChallenge format dataset into Scalabel format."""
import os
import os.path as osp
import shutil
import random
from collections import defaultdict
from typing import List, Tuple

import motmetrics as mm
from scalabel.label.from_mot import from_mot
from scalabel.label.io import load, load_label_config
from scalabel.label.typing import Dataset, Frame

from vis4d.struct import MetricLogs

from .base import BaseDatasetConfig, BaseDatasetLoader


class MOTDatasetConfig(BaseDatasetConfig):
    """Config for training/evaluation datasets."""

    tmp_dir: str = "./mot17_tmp"
    track_iou_thr: float = 0.5


class MOTChallenge(BaseDatasetLoader):
    """MOTChallenge dataloading class."""

    def __init__(self, cfg: BaseDatasetConfig):
        """Init dataset loader."""
        super().__init__(cfg)
        self.cfg: MOTDatasetConfig = MOTDatasetConfig(**cfg.dict())
        self.data_root = self.cfg.data_root
        if self.data_root.endswith(".hdf5"):  # pragma: no cover
            self.data_root = self.data_root.replace(".hdf5", "")
        self.tmp_dir = f"{self.cfg.tmp_dir}_{random.randint(0, 99999)}"

    def load_dataset(self) -> Dataset:  # pragma: no cover
        """Convert MOTChallenge annotations to scalabel format."""
        if self.cfg.annotations is None:
            dataset = from_mot(self.cfg.data_root)
        else:
            dataset = load(self.cfg.annotations)
        assert isinstance(dataset, Dataset)

        if self.cfg.config_path is not None:
            dataset.config = load_label_config(self.cfg.config_path)
        return dataset

    def _convert_predictions(
        self, frames: List[Frame]
    ) -> Tuple[List[str], List[str]]:
        """Convert predictions back to MOT format, save out to tmp_dir."""
        os.makedirs(self.tmp_dir, exist_ok=True)
        res_files = []
        frames_per_video = defaultdict(list)
        for f in frames:
            assert f.videoName is not None
            frames_per_video[f.videoName].append(f)

        for video, video_frames in frames_per_video.items():
            res_file = f"{self.tmp_dir}/res_{video}.txt"
            res_lines = ""
            for f in video_frames:
                if f.labels is not None:
                    for l in f.labels:
                        assert l.box2d is not None
                        x1, y1, x2, y2 = (
                            l.box2d.x1,
                            l.box2d.y1,
                            l.box2d.x2,
                            l.box2d.y2,
                        )
                        conf = l.score if l.score is not None else 1.0
                        assert f.frameIndex is not None and l.id is not None
                        res_lines += (
                            f"{f.frameIndex + 1},{l.id},{x1:.3f},{y1:.3f},"
                            f"{(x2 - x1):.3f},{(y2 - y1):.3f},{conf:.3f}\n"
                        )

            with open(res_file, "w", encoding="utf-8") as file:
                file.write(res_lines)
            res_files.append(res_file)

        return res_files, list(frames_per_video.keys())

    def _check_metrics(self) -> None:
        """Check if evaluation metrics specified are valid."""
        for metric in self.cfg.eval_metrics:
            if metric not in ["detect", "track"]:
                raise KeyError(
                    f"metric {metric} is not supported in {self.cfg.name}"
                )

    def evaluate(
        self, metric: str, predictions: List[Frame], gts: List[Frame]
    ) -> Tuple[MetricLogs, str]:
        """Evaluate according to MOT Challenge metrics."""
        if not metric == "track":  # pragma: no cover
            return super().evaluate(metric, predictions, gts)

        res_files, names = self._convert_predictions(predictions)
        accs = []
        for name, res_file in zip(names, res_files):
            gt_file = osp.join(self.data_root, f"{name}/gt/gt_half_val.txt")
            if not osp.exists(gt_file):
                raise FileNotFoundError(
                    "Couldn't find the GT file of the validation split!"
                )
            gt = mm.io.loadtxt(gt_file)
            res = mm.io.loadtxt(res_file)
            ini_file = osp.join(self.data_root, f"{name}/seqinfo.ini")
            if osp.exists(ini_file):
                acc, _ = mm.utils.CLEAR_MOT_M(
                    gt, res, ini_file, distth=1 - self.cfg.track_iou_thr
                )
            else:  # pragma: no cover
                acc = mm.utils.compare_to_groundtruth(
                    gt, res, distth=1 - self.cfg.track_iou_thr
                )
            accs.append(acc)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            names=names,
            metrics=mm.metrics.motchallenge_metrics,
            generate_overall=True,
        )
        str_summary = "\n" + mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names,
        )
        # clean up tmp dir
        shutil.rmtree(self.tmp_dir)
        log_dict = {k: v["OVERALL"] for k, v in summary.to_dict().items()}
        return log_dict, str_summary
