"""SHIFT result writer."""
from __future__ import annotations

import io
import itertools
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from vis4d.common import GenericFunc
from vis4d.common.typing import GenericFunc, NDArrayNumber
from vis4d.data.datasets.shift import shift_det_map
from vis4d.data.io import ZipBackend
from vis4d.eval.base import Writer

if SCALABEL_AVAILABLE:
    from scalabel.label.transforms import mask_to_rle, xyxy_to_box2d
    from scalabel.label.typing import Dataset, Frame, Label


class SHIFTWriter(Writer):
    """SHIFT result writer."""

    inverse_cat_map = {v: k for k, v in shift_det_map.items()}

    def __init__(
        self,
        output_dir: str,
        submission_file: str = "submission.zip",
    ) -> None:
        """Creates a new writer.

        Args:
            output_dir (str): Output directory.
        """
        super().__init__(output_dir, backend=ZipBackend())
        assert submission_file.endswith(
            ".zip"
        ), "Submission file must be a zip file."
        self.output_path = os.path.join(output_dir, submission_file)
        self.frames_det_2d = []
        self.frames_det_3d = []
        self.sample_counts: defaultdict = defaultdict(int)

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
        image = Image.fromarray(depth_map.astype("uint8"), mode="RGB")
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        self.backend.set(
            f"{self.output_path}/depth/{video_name}/{sample_name}",
            image_bytes.getvalue(),
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

    def process(
        self,
        frame_ids: list[int],
        sample_names: list[str],
        sequence_names: list[str],
        pred_sem_mask: list[NDArrayNumber] | None = None,
        pred_depth: list[NDArrayNumber] | None = None,
        pred_flow: list[NDArrayNumber] | None = None,
        pred_boxes2d: list[NDArrayNumber] | None = None,
        pred_boxes2d_classes: list[NDArrayNumber] | None = None,
        pred_boxes2d_scores: list[NDArrayNumber] | None = None,
        pred_boxes2d_track_ids: list[NDArrayNumber] | None = None,
        pred_instance_mask: list[NDArrayNumber] | None = None,
    ) -> None:
        """Process SHIFT results."""
        for i, (frame_id, sample_name, sequence_name) in enumerate(
            zip(frame_ids, sample_names, sequence_names)
        ):
            if pred_sem_mask is not None:
                self._write_sem_mask(
                    pred_sem_mask[i], sample_name, sequence_name
                )
                self.sample_counts["semseg"] += 1
            if pred_depth is not None:
                self._write_depth(pred_depth[i], sample_name, sequence_name)
                self.sample_counts["depth"] += 1
            if pred_flow is not None:
                self._write_flow(pred_flow[i], sample_name, sequence_name)
                self.sample_counts["flow"] += 1
            if pred_boxes2d is not None and pred_boxes2d_classes is not None:
                labels = []
                for box, score, class_id in zip(
                    pred_boxes2d[i],
                    pred_boxes2d_scores[i],
                    pred_boxes2d_classes[i],
                ):
                    box2d = xyxy_to_box2d(*box.tolist())
                    label = Label(
                        box2d=box2d,
                        category=self.inverse_cat_map[int(class_id)]
                        if self.inverse_cat_map != {}
                        else str(class_id),
                        score=float(score),
                        rle=mask_to_rle(
                            (pred_instance_mask[i][class_id] > 0.0).astype(np.uint8)  # type: ignore # pylint: disable=line-too-long
                        )
                        if pred_instance_mask
                        else None,
                        id=str(int(pred_boxes2d_track_ids[i][0]))
                        if pred_boxes2d_track_ids
                        else None,
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

    def gather(  # type: ignore # pragma: no cover
        self, gather_func: GenericFunc
    ) -> None:
        """Gather variables in case of distributed setting (if needed).

        Args:
            gather_func (Callable[[Any], Any]): Gather function.
        """
        all_preds = gather_func(self.frames_det_2d)
        if all_preds is not None:
            self.frames_det_2d = list(itertools.chain(*all_preds))

    def save(self) -> None:
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
            ds = Dataset(frames=self.frames_det_2d)
            ds_bytes = json.dumps(ds.dict()).encode("utf-8")
            self.backend.set(f"{self.output_path}/det_2d.json", ds_bytes)
