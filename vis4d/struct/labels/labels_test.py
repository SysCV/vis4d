"""Test cases for Vis4D annotation data structures."""
import unittest

import torch
from scalabel.label.typing import ImageSize

from vis4d.unittest.utils import (
    generate_dets,
    generate_dets3d,
    generate_instance_masks,
    generate_semantic_masks,
)

from ..data import Extrinsics
from .boxes import Boxes2D, Boxes3D
from .masks import MaskLogits, Masks, SemanticMasks


class TestBoxes2D(unittest.TestCase):
    """Test cases Vis4D Boxes2D."""

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

        for det, det_new in zip(detections, detections_new):
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
        self.assertEqual(detections.score, None)
        scalabel_dets_no_score = detections.to_scalabel(idx_to_class)
        self.assertTrue(all(d.score is None for d in scalabel_dets_no_score))

    def test_clone(self) -> None:
        """Testcase for cloning a Boxes2D object."""
        h, w, num_dets = 128, 128, 10
        detections = generate_dets(h, w, num_dets, track_ids=True)
        detections_new = detections.clone()

        for det, det_new in zip(detections, detections_new):
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
    """Test cases Vis4D Boxes3D."""

    def test_transform(self) -> None:
        """Test transform function."""
        num_dets = 10
        detections = generate_dets3d(num_dets, track_ids=True)
        # test transform
        # pylint: disable=no-member
        dets_tr = detections.clone()
        dets_tr.transform(Extrinsics(torch.eye(4).unsqueeze(0)))
        self.assertTrue(
            torch.isclose(dets_tr.boxes, detections.boxes, atol=1e-4).all()
        )

        box3d = Boxes3D(
            torch.tensor(
                [
                    [
                        18.63882979851619,
                        0.19359276352412746,
                        59.024867320654835,
                        1.642,
                        0.621,
                        0.669,
                        0,
                        -3.120828115749234,
                        0,
                    ]
                ]
            )
        )
        extrinsics = Extrinsics(
            torch.tensor(
                [
                    [
                        -9.40165212e-01,
                        -1.55825481e-02,
                        -3.40362392e-01,
                        4.10872441e02,
                    ],
                    [
                        3.39996835e-01,
                        2.20946316e-02,
                        -9.40166996e-01,
                        1.17957081e03,
                    ],
                    [
                        2.21703791e-02,
                        -9.99634439e-01,
                        -1.54745870e-02,
                        1.49367752e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        1.00000000e00,
                    ],
                ]
            )
        )
        box3d_worldspace = torch.tensor(
            [[373.26, 1130.42, 0.80, 1.64, 0.62, 0.67, -0.37]]
        )

        box3d_orig = box3d.clone()
        box3d.transform(extrinsics)

        # euler angles can differ here depending on the transformation
        self.assertTrue(
            torch.isclose(
                box3d.center, box3d_worldspace[:, :3], atol=1e-2
            ).all()
        )

        box3d.transform(extrinsics.inverse())
        self.assertTrue(
            torch.isclose(box3d.boxes, box3d_orig.boxes, atol=1e-2).all()
        )

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

        for det, det_new in zip(detections, detections_new):
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

        # without score
        detections.boxes = detections.boxes[:, :-1]
        self.assertEqual(detections.score, None)
        scalabel_dets_no_score = detections.to_scalabel(idx_to_class)
        self.assertTrue(all(d.score is None for d in scalabel_dets_no_score))

    def test_clone(self) -> None:
        """Testcase for cloning a Boxes2D object."""
        num_dets = 10
        detections = generate_dets3d(num_dets, track_ids=True)
        detections_new = detections.clone()

        for det, det_new in zip(detections, detections_new):
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


class TestMasks(unittest.TestCase):
    """Test cases Vis4D Masks."""

    def test_scalabel(self) -> None:
        """Testcase for conversion to / from scalabel."""
        h, w, num_masks = 128, 128, 10
        segmentations = generate_instance_masks(
            h, w, num_masks, track_ids=True
        )
        self.assertEqual(segmentations.height, 128)
        self.assertEqual(segmentations.width, 128)
        idx_to_class = {0: "car", 1: "background"}
        class_to_idx = {"car": 0, "background": 1}
        scalabel_segms = segmentations.to_scalabel(idx_to_class)

        segms_new = Masks.from_scalabel(
            scalabel_segms,
            class_to_idx,
            image_size=ImageSize(width=w, height=h),
            bg_as_class=True,
        )

        scalabel_segms[0].rle = None
        segms_with_none = Masks.from_scalabel(
            scalabel_segms,
            class_to_idx,
            image_size=ImageSize(width=w, height=h),
            bg_as_class=True,
        )
        self.assertTrue(
            torch.isclose(segms_with_none.masks[0], segms_new.masks[1]).all()
        )

        for segm, segm_new in zip(segmentations, segms_new):
            self.assertTrue(torch.isclose(segm.masks, segm_new.masks).all())
            self.assertTrue(
                torch.isclose(segm.class_ids.long(), segm_new.class_ids).all()
            )
            self.assertTrue(
                torch.isclose(segm.track_ids.long(), segm_new.track_ids).all()
            )
            self.assertTrue(
                torch.isclose(segm.score.float(), segm_new.score).all()
            )

        segms_new.track_ids = None
        segms_without_tracks = segms_new.to_scalabel(idx_to_class)
        self.assertTrue(
            all(
                (
                    str(i) == segm.id
                    for i, segm in enumerate(segms_without_tracks)
                )
            )
        )

        segmentations.score = None
        scalabel_segms_no_score = segmentations.to_scalabel(idx_to_class)
        self.assertTrue(all(d.score is None for d in scalabel_segms_no_score))

    def test_clone(self) -> None:
        """Testcase for cloning a Masks object."""
        h, w, num_masks = 128, 128, 10
        segmentations = generate_instance_masks(
            h, w, num_masks, track_ids=True
        )
        segms_new = segmentations.clone()

        for segm, segm_new in zip(segmentations, segms_new):
            self.assertTrue(torch.isclose(segm.masks, segm_new.masks).all())
            self.assertTrue(
                torch.isclose(segm.class_ids, segm_new.class_ids).all()
            )
            self.assertTrue(
                torch.isclose(segm.track_ids, segm_new.track_ids).all()
            )

    def test_resize(self) -> None:
        """Testcase for resizing a Masks object."""
        h, w, num_masks = 128, 128, 10
        segmentations = generate_instance_masks(
            h, w, num_masks, track_ids=True
        )
        segmentations.resize((64, 256))
        self.assertEqual(segmentations.height, 256)
        self.assertEqual(segmentations.width, 64)

    def test_from_hwc(self) -> None:
        """Testcase for conversion from a HxW semantic mask."""
        masks = SemanticMasks.from_hwc_tensor(
            torch.Tensor([[0, 1], [0, 255]]).int()
        )
        self.assertEqual(len(masks.masks), 2)
        self.assertEqual(len(masks.masks.unique()), 2)

    def test_semantic_postprocess(self) -> None:
        """Testcase for postprocessing a Semantic Masks object."""
        h, w, num_masks = 128, 128, 10
        segmentations = generate_semantic_masks(h, w, num_masks)
        segmentations.postprocess((w - 1, h - 1), (w - 1, h - 1))
        self.assertEqual(segmentations.height, 127)
        self.assertEqual(segmentations.width, 127)
        segmentations.postprocess((64, 256), (w - 2, h - 2))
        self.assertEqual(segmentations.height, 256)
        self.assertEqual(segmentations.width, 64)

    def test_crop_and_resize(self) -> None:
        """Testcase for cropping and resizing a Masks object."""
        h, w, num_masks, num_dets = 128, 128, 4, 10
        out_h, out_w = 64, 32
        inds = torch.LongTensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
        segmentations = generate_instance_masks(
            h, w, num_masks, track_ids=True
        )[inds]
        detections = generate_dets(h, w, num_dets, track_ids=True)
        segm_crops = segmentations.crop_and_resize(detections, (out_h, out_w))
        self.assertEqual(len(segm_crops.masks), num_dets)
        self.assertEqual(segm_crops.masks.size(1), out_h)
        self.assertEqual(segm_crops.masks.size(2), out_w)
        segm_crops = segmentations.crop_and_resize(
            detections, (out_h, out_w), binarize=False
        )
        self.assertEqual(len(segm_crops.masks), num_dets)
        self.assertEqual(segm_crops.masks.size(1), out_h)
        self.assertEqual(segm_crops.masks.size(2), out_w)
        segmentations = generate_instance_masks(h, w, 0, track_ids=True)
        detections = generate_dets(h, w, 0, track_ids=True)
        segm_crops = segmentations.crop_and_resize(detections, (out_h, out_w))
        self.assertEqual(len(segm_crops), 0)

    def test_get_boxes2d(self) -> None:
        """Testcase for get_boxes2d function."""
        h, w, num_masks = 28, 28, 5
        segmentations = generate_instance_masks(
            h, w, num_masks, track_ids=True
        )
        boxes = segmentations.get_boxes2d()
        self.assertEqual(len(boxes), len(segmentations))
        self.assertTrue(torch.isclose(boxes.score, segmentations.score).all())
        segmentations = generate_instance_masks(h, w, 0, track_ids=True)
        boxes = segmentations.get_boxes2d()
        self.assertEqual(len(boxes), 0)

    def test_instance_postprocess(self) -> None:
        """Testcase for postprocessing a Instance Masks object."""
        h, w, num_masks, num_dets = 28, 28, 10, 10
        segmentations = generate_instance_masks(h, w, num_masks)
        dets = generate_dets(h, w, num_dets)
        segmentations.detections = dets
        segmentations.postprocess((w - 1, h - 1), (w - 2, h - 2))
        self.assertEqual(segmentations.height, 27)
        self.assertEqual(segmentations.width, 27)

    def test_pad(self) -> None:
        """Testcase for pad function."""
        h, w, num_masks = 28, 28, 5
        pad_shape = (56, 128)
        segmentations = generate_semantic_masks(h, w, num_masks)
        segmentations = SemanticMasks.pad([segmentations], pad_shape)[0]
        self.assertEqual(segmentations.height, 128)
        self.assertEqual(segmentations.width, 56)
        segmentations = SemanticMasks.pad([segmentations], pad_shape)[0]
        self.assertEqual(segmentations.height, 128)
        self.assertEqual(segmentations.width, 56)

    def test_mask_logits(self) -> None:
        """Testcase for MaskLogits."""
        h, w, num_masks, num_dets = 28, 28, 5, 5
        pad_shape = (56, 128)
        logits = MaskLogits(torch.rand((num_masks, h, w)))
        logits.resize(pad_shape)
        self.assertEqual(logits.size, pad_shape)

        logits = MaskLogits(torch.rand((num_masks, h, w)))
        dets = generate_dets(h, w, num_dets)
        logits_paste = logits.paste_masks(dets, pad_shape)
        self.assertEqual(logits_paste.size, pad_shape)
        self.assertEqual(len(logits_paste), num_masks)

        logits = MaskLogits(torch.rand((num_masks, h, w)))
        logits.postprocess(pad_shape, (w, h))
        self.assertEqual(logits.size, pad_shape)
