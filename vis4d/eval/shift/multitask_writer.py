"""SHIFT result writer."""

from __future__ import annotations

import io
import itertools
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from vis4d.common import MetricLogs
from vis4d.common.array import array_to_numpy
from vis4d.common.imports import SCALABEL_AVAILABLE
from vis4d.common.typing import ArrayLike, GenericFunc, NDArrayNumber
from vis4d.data.datasets.shift import shift_det_map
from vis4d.data.io import DataBackend, ZipBackend
from vis4d.eval.base import Evaluator

if SCALABEL_AVAILABLE:
    from scalabel.label.transforms import mask_to_rle, xyxy_to_box2d
    from scalabel.label.typing import Dataset, Frame, Label
else:
    raise ImportError("scalabel is not installed.")


class SHIFTMultitaskWriter(Evaluator):
    """SHIFT result writer for online evaluation."""

    inverse_cat_map = {v: k for k, v in shift_det_map.items()}

    def __init__(
        self,
        output_dir: str,
        submission_file: str = "submission.zip",
    ) -> None:
        """Creates a new writer.

        Args:
            output_dir (str): Output directory.
            submission_file (str): Submission file name. Defaults to
                "submission.zip".
        """
        super().__init__()
        assert submission_file.endswith(
            ".zip"
        ), "Submission file must be a zip file."
        self.backend: DataBackend = ZipBackend()
        self.output_path = os.path.join(output_dir, submission_file)
        self.frames_det_2d: list[Frame] = []
        self.frames_det_3d: list[Frame] = []
        self.sample_counts: defaultdict[str, int] = defaultdict(int)

    def _write_sem_mask(
        self, sem_mask: NDArrayNumber, sample_name: str, video_name: str
    ) -> None:
        """Write semantic mask.

        Args:
            sem_mask (NDArrayNumber): Predicted semantic mask, shape (H, W).
            sample_name (str): Sample name.
            video_name (str): Video name.
        """
        image = Image.fromarray(sem_mask.astype("uint8"), mode="L")
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        self.backend.set(
            f"{self.output_path}/semseg/{video_name}/{sample_name}",
            image_bytes.getvalue(),
            mode="w",
        )

    def _write_depth(
        self, depth_map: NDArrayNumber, sample_name: str, video_name: str
    ) -> None:
        """Write depth map.

        Args:
            depth_map (NDArrayNumber): Predicted depth map, shape (H, W).
            sample_name (str): Sample name.
            video_name (str): Video name.
        """
        depth_map = np.clip(depth_map / 80.0 * 255.0, 0, 255)
        image = Image.fromarray(depth_map.astype("uint8"), mode="L")
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        self.backend.set(
            f"{self.output_path}/depth/{video_name}/{sample_name}",
            image_bytes.getvalue(),
            mode="w",
        )

    def _write_flow(
        self, flow: NDArrayNumber, sample_name: str, video_name: str
    ) -> None:
        """Write semantic mask.

        Args:
            flow (NDArrayNumber): Predicted optical flow, shape (H, W, 2).
            sample_name (str): Sample name.
            video_name (str): Video name.
        """
        raise NotImplementedError

    def process_batch(
        self,
        frame_ids: list[int],
        sample_names: list[str],
        sequence_names: list[str],
        pred_sem_mask: list[ArrayLike] | None = None,
        pred_depth: list[ArrayLike] | None = None,
        pred_flow: list[ArrayLike] | None = None,
        pred_boxes2d: list[ArrayLike] | None = None,
        pred_boxes2d_classes: list[ArrayLike] | None = None,
        pred_boxes2d_scores: list[ArrayLike] | None = None,
        pred_boxes2d_track_ids: list[ArrayLike] | None = None,
        pred_instance_masks: list[ArrayLike] | None = None,
    ) -> None:
        """Process SHIFT results.

        You can omit some of the predictions if they are not used.

        Args:
            frame_ids (list[int]): Frame IDs.
            sample_names (list[str]): Sample names.
            sequence_names (list[str]): Sequence names.
            pred_sem_mask (list[ArrayLike], optional): Predicted semantic
                masks, each in shape (C, H, W) or (H, W). Defaults to None.
            pred_depth (list[ArrayLike], optional): Predicted depth maps,
                each in shape (H, W), with meter unit. Defaults to None.
            pred_flow (list[ArrayLike], optional): Predicted optical flows,
                each in shape (H, W, 2). Defaults to None.
            pred_boxes2d (list[ArrayLike], optional): Predicted 2D boxes,
                each in shape (N, 4). Defaults to None.
            pred_boxes2d_classes (list[ArrayLike], optional): Predicted
                2D box classes, each in shape (N,). Defaults to None.
            pred_boxes2d_scores (list[ArrayLike], optional): Predicted
                2D box scores, each in shape (N,). Defaults to None.
            pred_boxes2d_track_ids (list[ArrayLike], optional): Predicted
                2D box track IDs, each in shape (N,). Defaults to None.
            pred_instance_masks (list[ArrayLike], optional): Predicted
                instance masks, each in shape (N, H, W). Defaults to None.
        """
        for i, (frame_id, sample_name, sequence_name) in enumerate(
            zip(frame_ids, sample_names, sequence_names)
        ):
            if pred_sem_mask is not None:
                sem_mask_ = array_to_numpy(
                    pred_sem_mask[i],
                    n_dims=None,
                    dtype=np.float32,
                )
                if len(sem_mask_.shape) == 3:
                    sem_mask = sem_mask_.argmax(axis=0)
                else:
                    sem_mask = sem_mask_.astype(np.uint8)
                semseg_filename = sample_name.replace(".jpg", ".png").replace(
                    "img", "semseg"
                )
                self._write_sem_mask(sem_mask, semseg_filename, sequence_name)
                self.sample_counts["semseg"] += 1
            if pred_depth is not None:
                depth = array_to_numpy(
                    pred_depth[i], n_dims=None, dtype=np.float32
                )
                depth_filename = sample_name.replace(".jpg", ".png").replace(
                    "img", "depth"
                )
                self._write_depth(depth, depth_filename, sequence_name)
                self.sample_counts["depth"] += 1
            if pred_flow is not None:
                flow = array_to_numpy(
                    pred_flow[i], n_dims=None, dtype=np.float32
                )
                self._write_flow(flow, sample_name, sequence_name)
                self.sample_counts["flow"] += 1
            if (
                pred_boxes2d is not None
                and pred_boxes2d_classes is not None
                and pred_boxes2d_scores is not None
            ):
                labels = []
                if pred_instance_masks:
                    masks = array_to_numpy(
                        pred_instance_masks[i], n_dims=None, dtype=np.float32
                    )
                if pred_boxes2d_track_ids:
                    track_ids = array_to_numpy(
                        pred_boxes2d_track_ids[i],
                        n_dims=None,
                        dtype=np.int64,
                    )
                for box, score, class_id in zip(
                    pred_boxes2d[i],
                    pred_boxes2d_scores[i],
                    pred_boxes2d_classes[i],
                ):
                    box2d = xyxy_to_box2d(*box.tolist())
                    if pred_instance_masks:
                        rle = mask_to_rle(
                            (masks[class_id] > 0.0).astype(np.uint8)
                        )
                    else:
                        rle = None

                    if pred_boxes2d_track_ids:
                        track_id = str(int(track_ids[0]))
                    else:
                        track_id = None

                    label = Label(
                        box2d=box2d,
                        category=(
                            self.inverse_cat_map[int(class_id)]
                            if self.inverse_cat_map != {}
                            else str(class_id)
                        ),
                        score=float(score),
                        rle=rle,
                        id=track_id,
                    )
                    labels.append(label)
                frame = Frame(
                    name=sample_name,
                    videoName=sequence_name,
                    frameIndex=frame_id,
                    labels=labels,
                )
                self.frames_det_2d.append(frame)
                self.sample_counts["det_2d"] += 1

    def gather(self, gather_func: GenericFunc) -> None:  # pragma: no cover
        """Gather variables in case of distributed setting (if needed).

        Args:
            gather_func (Callable[[Any], Any]): Gather function.
        """
        all_preds = gather_func(self.frames_det_2d)
        if all_preds is not None:
            self.frames_det_2d = list(itertools.chain(*all_preds))

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """No evaluation locally."""
        return {}, "No evaluation locally."

    def save(self, metric: str, output_dir: str) -> None:
        """Save scalabel output to zip file.

        Raises:
            ValueError: If the number of samples in each category is not the
                same.
        """
        # Check if the sample counts are correct
        equal_size = True
        for key in self.sample_counts:
            if self.sample_counts[key] != len(self.frames_det_2d):
                equal_size = False
                break
        if not equal_size:
            raise ValueError(
                "The number of samples in each category is not the same."
            )

        # Save the 2D detection results
        if len(self.frames_det_2d) > 0:
            ds = Dataset(frames=self.frames_det_2d, groups=None, config=None)
            ds_bytes = json.dumps(ds.dict()).encode("utf-8")
            self.backend.set(
                f"{self.output_path}/det_2d.json", ds_bytes, mode="w"
            )

        self.backend.close()
        print(f"Saved the submission file at {self.output_path}.")
