"""Test cases for VisT annotation data structures."""
import unittest

import torch

from vist.struct import Boxes2D, Boxes3D
from vist.unittest.utils import generate_dets, generate_dets3d


class TestBoxes2D(unittest.TestCase):
    """Test cases VisT Boxes2D."""

    def test_scalabel(self) -> None:
        """Testcase for conversion to / from scalabel."""
        h, w, num_dets = 128, 128, 10
        detections = generate_dets(h, w, num_dets, track_ids=True)
        idx_to_class = {0: "car"}
        class_to_idx = {"car": 0}
        scalabel_dets = detections.to_scalabel(idx_to_class)

        detections_new = Boxes2D.from_scalabel(scalabel_dets, class_to_idx)

        scalabel_dets[0].box2d = None
        dets_with_none = Boxes2D.from_scalabel(scalabel_dets, class_to_idx)
        self.assertTrue(
            torch.isclose(
                dets_with_none.boxes[0], detections_new.boxes[1]
            ).all()
        )

        for det, det_new in zip(detections, detections_new):  # type: ignore
            self.assertTrue(torch.isclose(det.boxes, det_new.boxes).all())
            self.assertTrue(
                torch.isclose(det.class_ids.long(), det_new.class_ids).all()
            )
            self.assertTrue(
                torch.isclose(det.track_ids.long(), det_new.track_ids).all()
            )

        detections_new.track_ids = None
        dets_without_tracks = detections_new.to_scalabel(idx_to_class)
        self.assertTrue(
            all(
                (str(i) == det.id for i, det in enumerate(dets_without_tracks))
            )
        )

        detections.boxes = detections.boxes[:, :-1]
        scalabel_dets_no_score = detections.to_scalabel(idx_to_class)
        self.assertTrue(all(d.score is None for d in scalabel_dets_no_score))

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

    def test_merge(self) -> None:
        """Testcase for merging a list of Boxes2D objects."""
        h, w, num_dets = 128, 128, 10
        det = generate_dets(h, w, num_dets, track_ids=True)
        det_new = Boxes2D.merge([det, det])

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


class TestBoxes3D(unittest.TestCase):
    """Test cases VisT Boxes3D."""

    def test_scalabel(self) -> None:
        """Testcase for conversion to / from scalabel."""
        num_dets = 10
        detections = generate_dets3d(num_dets, track_ids=True)
        idx_to_class = {0: "car"}
        class_to_idx = {"car": 0}
        scalabel_dets = detections.to_scalabel(idx_to_class)

        detections_new = Boxes3D.from_scalabel(scalabel_dets, class_to_idx)

        scalabel_dets[0].box3d = None
        dets_with_none = Boxes3D.from_scalabel(scalabel_dets, class_to_idx)
        self.assertTrue(
            torch.isclose(
                dets_with_none.boxes[0], detections_new.boxes[1]
            ).all()
        )

        for det, det_new in zip(detections, detections_new):  # type: ignore
            self.assertTrue(torch.isclose(det.boxes, det_new.boxes).all())
            self.assertTrue(
                torch.isclose(det.class_ids.long(), det_new.class_ids).all()
            )
            self.assertTrue(
                torch.isclose(det.track_ids.long(), det_new.track_ids).all()
            )

        detections_new.track_ids = None
        dets_without_tracks = detections_new.to_scalabel(idx_to_class)
        self.assertTrue(
            all(
                (str(i) == det.id for i, det in enumerate(dets_without_tracks))
            )
        )

        rotx, roty, rotz = detections.rot_x, detections.rot_y, detections.rot_z
        self.assertIsNotNone(rotx)
        self.assertIsNotNone(roty)
        self.assertIsNotNone(rotz)

        # test 7 DoF
        detections_7dof = detections.clone()
        detections_7dof.boxes = detections_7dof.boxes[
            :, [0, 1, 2, 3, 4, 5, 7, 9]
        ]
        scalabel_dets_7dof = detections_7dof.to_scalabel(  # pylint: disable=no-member,line-too-long
            idx_to_class
        )
        for d in scalabel_dets_7dof:
            assert d.box3d is not None
            self.assertEqual(d.box3d.orientation[0], 0)
            self.assertEqual(d.box3d.orientation[2], 0)

        rotx, roty, rotz = (
            detections_7dof.rot_x,  # pylint: disable=no-member
            detections_7dof.rot_y,  # pylint: disable=no-member
            detections_7dof.rot_z,  # pylint: disable=no-member
        )
        self.assertIsNone(rotx)
        self.assertIsNotNone(roty)
        self.assertIsNone(rotz)

        # 7DoF without score
        detections_7dof.boxes = detections_7dof.boxes[:, :-1]
        scalabel_dets_no_score = detections_7dof.to_scalabel(  # pylint: disable=no-member,line-too-long
            idx_to_class
        )
        self.assertIsNone(detections_7dof.score)  # pylint: disable=no-member
        for d in scalabel_dets_no_score:
            self.assertIsNone(d.score)
            assert d.box3d is not None
            self.assertEqual(d.box3d.orientation[0], 0)
            self.assertEqual(d.box3d.orientation[2], 0)

        # without score
        detections.boxes = detections.boxes[:, :-1]
        scalabel_dets_no_score = detections.to_scalabel(idx_to_class)
        self.assertTrue(all(d.score is None for d in scalabel_dets_no_score))

    def test_clone(self) -> None:
        """Testcase for cloning a Boxes2D object."""
        num_dets = 10
        detections = generate_dets3d(num_dets, track_ids=True)
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
        num_dets = 10
        det = generate_dets3d(num_dets, track_ids=True)
        det_new = Boxes3D.merge([det, det])

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
