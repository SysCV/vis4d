"""Load MOTChallenge format dataset into Scalabel format."""
import os
import os.path as osp
import shutil
import tempfile
from collections import defaultdict
from typing import List, Optional, Tuple

import motmetrics as mm
from scalabel.label.from_mot import from_mot
from scalabel.label.io import load
from scalabel.label.transforms import box2d_to_xyxy
from scalabel.label.typing import Dataset, Frame

from vis4d.struct import ArgsType, MetricLogs

from .base import BaseDatasetLoader


class MOTChallenge(BaseDatasetLoader):
    """MOTChallenge dataloading class."""

    def __init__(
        self,
        *args: ArgsType,
        tmp_dir_root: str = "./",
        track_iou_thr: float = 0.5,
        gt_root: Optional[str] = None,
        det_metrics_per_video: bool = False,
        **kwargs: ArgsType,
    ):
        """Init dataset loader."""
        super().__init__(*args, **kwargs)
        self.det_metrics_per_video = det_metrics_per_video
        self.track_iou_thr = track_iou_thr
        self.tmp_dir_root = tmp_dir_root
        if gt_root is None:
            self.gt_root = self.data_root
        else:
            self.gt_root = self.gt_root  # pragma: no cover

    def load_dataset(self) -> Dataset:  # pragma: no cover
        """Convert MOTChallenge annotations to scalabel format."""
        if self.annotations is None:
            dataset = from_mot(self.data_root)
        else:
            dataset = load(self.annotations)
        assert isinstance(dataset, Dataset)

        if self.config_path is not None:
            _, metadata_cfg = self.load_config()
            dataset.config = metadata_cfg
        return dataset

    def _convert_predictions(
        self, frames: List[Frame]
    ) -> Tuple[List[str], List[str], str]:
        """Convert predictions back to MOT format, save out to tmp_dir."""
        os.makedirs(self.tmp_dir_root, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(dir=self.tmp_dir_root)
        res_files = []
        frames_per_video = defaultdict(list)
        for f in frames:
            assert f.videoName is not None
            frames_per_video[f.videoName].append(f)

        for video, video_frames in frames_per_video.items():
            res_file = f"{tmp_dir}/res_{video}.txt"
            res_lines = ""
            for f in video_frames:
                if f.labels is not None:
                    for l in f.labels:
                        assert l.box2d is not None
                        x1, y1, x2, y2 = box2d_to_xyxy(l.box2d)
                        conf = l.score if l.score is not None else 1.0
                        assert f.frameIndex is not None and l.id is not None
                        res_lines += (
                            f"{f.frameIndex + 1},{l.id},{x1:.3f},{y1:.3f},"
                            f"{(x2 - x1):.3f},{(y2 - y1):.3f},{conf:.3f}\n"
                        )

            with open(res_file, "w", encoding="utf-8") as file:
                file.write(res_lines)
            res_files.append(res_file)

        return res_files, list(frames_per_video.keys()), tmp_dir

    def _check_metrics(self) -> None:
        """Check if evaluation metrics specified are valid."""
        assert self.eval_metrics is not None
        for metric in self.eval_metrics:
            if metric not in ["detect", "track"]:
                raise KeyError(
                    f"metric {metric} is not supported in {self.name}"
                )

    def evaluate(
        self, metric: str, predictions: List[Frame], gts: List[Frame]
    ) -> Tuple[MetricLogs, str]:
        """Evaluate according to MOT Challenge metrics."""
        if not metric == "track":  # pragma: no cover
            log_dict, log_str = super().evaluate(metric, predictions, gts)
            if self.det_metrics_per_video:
                # per video detection results
                video_names = sorted(
                    set(f.videoName for f in gts if f.videoName is not None)
                )
                for video_name in video_names:
                    vid_dict, vid_str = super().evaluate(
                        metric,
                        [f for f in predictions if f.videoName == video_name],
                        [f for f in gts if f.videoName == video_name],
                    )
                    for k in vid_dict:
                        log_dict[f"{k}/{video_name}"] = vid_dict[k]
                    log_str += f"\n{video_name}:{vid_str}"
            return log_dict, log_str

        res_files, names, tmp_dir = self._convert_predictions(predictions)
        accs = []
        for name, res_file in zip(names, res_files):
            gt_file = osp.join(self.gt_root, f"{name}/gt/gt_half_val.txt")
            if not osp.exists(gt_file):
                raise FileNotFoundError(
                    "Couldn't find the GT file of the validation split!"
                )
            gt = mm.io.loadtxt(gt_file)
            res = mm.io.loadtxt(res_file)
            ini_file = osp.join(self.gt_root, f"{name}/seqinfo.ini")
            if osp.exists(ini_file):
                acc, _ = mm.utils.CLEAR_MOT_M(
                    gt, res, ini_file, distth=1 - self.track_iou_thr
                )
            else:  # pragma: no cover
                acc = mm.utils.compare_to_groundtruth(
                    gt, res, distth=1 - self.track_iou_thr
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
        shutil.rmtree(tmp_dir)
        log_dict = {k: v["OVERALL"] for k, v in summary.to_dict().items()}
        return log_dict, str_summary
