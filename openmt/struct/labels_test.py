"""Test cases for openMT annotation data structures."""
import unittest

import torch

from openmt.struct import Boxes2D
from openmt.unittest.utils import generate_dets


class TestAnnotationStructures(unittest.TestCase):
    """Test cases openMT annotation data structures."""

    def test_scalabel(self) -> None:
        """Testcase for conversion to / from scalabel."""
        h, w, num_dets = 128, 128, 10
        detections = generate_dets(h, w, num_dets, track_ids=True)
        idx_to_class = {0: "car"}
        class_to_idx = {"car": 0}
        scalabel_dets = detections.to_scalabel(idx_to_class)

        detections_new = Boxes2D.from_scalabel(scalabel_dets, class_to_idx)

        scalabel_dets[0].box_2d = None
        dets_with_none = Boxes2D.from_scalabel(scalabel_dets, class_to_idx)
        self.assertTrue(
            torch.isclose(
                dets_with_none.boxes[0], detections_new.boxes[1]
            ).all()
        )

        for det, det_new in zip(detections, detections_new):  # type: ignore
            self.assertTrue(torch.isclose(det.boxes, det_new.boxes).all())
            self.assertTrue(
                torch.isclose(det.class_ids.int(), det_new.class_ids).all()
            )
            self.assertTrue(
                torch.isclose(det.track_ids.int(), det_new.track_ids).all()
            )

        detections_new.track_ids = None
        dets_without_tracks = detections_new.to_scalabel(idx_to_class)
        self.assertTrue(
            all(
                (str(i) == det.id for i, det in enumerate(dets_without_tracks))
            )
        )

    def test_clone(self) -> None:
        """Testcase for cloning a Boxes2D object."""
        h, w, num_dets = 128, 128, 10
        detections = generate_dets(h, w, num_dets, track_ids=True)
        detections_new = detections.clone()

        for det, det_new in zip(detections, detections_new):  # type: ignore
            self.assertTrue(torch.isclose(det.boxes, det_new.boxes).all())
            self.assertTrue(
                torch.isclose(det.class_ids, det_new.class_ids).all()
            )
            self.assertTrue(
                torch.isclose(det.track_ids, det_new.track_ids).all()
            )

    def test_cat(self) -> None:
        """Testcase for concatenating a list of Boxes2D objects."""
        h, w, num_dets = 128, 128, 10
        det = generate_dets(h, w, num_dets, track_ids=True)
        det_new = Boxes2D.cat([det, det])

        self.assertTrue(
            torch.isclose(
                torch.cat([det.boxes, det.boxes]), det_new.boxes
            ).all()
        )

        self.assertTrue(
            torch.isclose(
                torch.cat([det.class_ids, det.class_ids]), det_new.class_ids
            ).all()
        )

        self.assertTrue(
            torch.isclose(
                torch.cat([det.track_ids, det.track_ids]), det_new.track_ids
            ).all()
        )
