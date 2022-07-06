"""Test cases for QD3DT tracking graph construction."""
import unittest

import torch
from scalabel.label.typing import Frame

from vis4d.model.track.graph.deepsort import DeepSORTTrackGraph
from vis4d.struct import Images, InputSample, LabelInstances
from vis4d.unittest.utils import generate_dets


def get_input_sample(frame_index: int):
    """ " Return an empty input sample for a given frame index"""
    return InputSample(
        [Frame(name="myframe", frameIndex=frame_index)],
        Images(torch.empty(1, 128, 128, 3), image_sizes=[(128, 128)]),
    )


class TestDeepSortTrack(unittest.TestCase):
    """Test cases for QD3DT tracking graph construction."""

    min_confidence = 0.4
    num_classes = 4
    max_age = 5
    tracker = DeepSORTTrackGraph(
        num_classes=num_classes,
        kalman_filter_params={
            "cov_motion_Q": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "cov_project_R": [0.1, 0.1, 0.1, 0.1],
            "cov_P0": [1, 1, 1, 1, 1, 1, 1, 1],
        },
        min_confidence=min_confidence,
        max_age=max_age,
    )

    def test_get_tracks(self) -> None:
        """Testcase for get tracks method."""
        self.tracker.reset()
        # Expect tracks to be empty at beginning
        self.assertTrue(len(self.tracker.get_tracks()) == 0)

        h, w, num_dets = 128, 128, 64
        # Create random detections all with same class (0)
        detections = LabelInstances([generate_dets(h, w, num_dets)])
        embeddings = torch.rand(num_dets, 128)

        # feed same detections & embeddings --> should be matched to self
        result_t0 = self.tracker(
            get_input_sample(frame_index=0), detections, embeddings=embeddings
        ).boxes2d[0]
        result_t1 = self.tracker(
            get_input_sample(frame_index=1), detections, embeddings=embeddings
        ).boxes2d[0]
        result_t2 = self.tracker(
            get_input_sample(frame_index=2), detections, embeddings=embeddings
        ).boxes2d[0]

        # check if matching is correct
        t0, t1, t2 = (
            result_t0.track_ids.sort()[0],
            result_t1.track_ids.sort()[0],
            result_t2.track_ids.sort()[0],
        )
        self.assertTrue(
            torch.sum(detections.boxes2d[0].score > self.min_confidence)
            == len(t0)
        )
        self.assertTrue((t0 == t1).all() and (t1 == t2).all())
        print(result_t2)

        # check if all tracks have scores >= threshold
        for res in [result_t0, result_t1, result_t2]:
            self.assertTrue((res.boxes[:, -1] >= self.min_confidence).all())

    def test_few_matches(self) -> None:
        """Testcase for get tracks method."""
        self.tracker.reset()
        # Expect tracks to be empty at beginning
        self.assertTrue(len(self.tracker.get_tracks()) == 0)

        h, w, num_dets = 128, 128, 64
        generated_detections = [generate_dets(h, w, num_dets)]
        # Create random detections all with same class (0)
        detections = LabelInstances(generated_detections)
        embeddings = torch.rand(num_dets, 128)
        self.tracker(
            get_input_sample(frame_index=0), detections, embeddings=embeddings
        )

        # Only confirm first 20 detections
        first_detections = LabelInstances([generated_detections[0][:20]])
        first_embeddings = embeddings[:20, :]

        result_t1 = self.tracker(
            get_input_sample(frame_index=1),
            first_detections,
            embeddings=first_embeddings,
        ).boxes2d[0]
        # Expect to only have the redetected matches now
        self.assertTrue(
            torch.sum(first_detections.boxes2d[0].score > self.min_confidence)
            == len(result_t1)
        )

    def test_new_detections(self) -> None:
        """Testcase for get tracks method."""
        self.tracker.reset()
        # Expect tracks to be empty at beginning
        self.assertTrue(len(self.tracker.get_tracks()) == 0)

        h, w, num_dets = 128, 128, 64
        generated_detections = [generate_dets(h, w, num_dets)]
        # Create random detections all with same class (0)
        detections = LabelInstances(generated_detections)
        embeddings = torch.rand(num_dets, 128)

        valid_detections = torch.sum(
            detections.boxes2d[0].score > self.min_confidence
        )
        # confirm initial tracks. If this is not done, not samples
        # will be returned as they have not been confirmed
        for i in range(4):
            result_t0 = self.tracker(
                get_input_sample(frame_index=i),
                detections,
                embeddings=embeddings,
            ).boxes2d[0]
        self.assertTrue(valid_detections == len(result_t0))

        # Unmatch first 30 detections
        for box in generated_detections[0][:30]:
            box.boxes[:, [0, 2]] += 50
            box.boxes[:, [1, 3]] += 50

        embeddings[:30, :] = torch.rand(
            30, 128
        )

        # create new features for first 30 detections
        result_t1 = self.tracker(
            get_input_sample(frame_index=4), detections, embeddings=embeddings
        ).boxes2d[0]
        self.assertTrue(
            (
                valid_detections
                + torch.sum(
                    generated_detections[0][:30].score > self.min_confidence
                )
            )
            == len(result_t1)
        )

        # Update with new detections and make sure boxes are matched and
        # updated
        for i in range(5, 5 + self.max_age + 2):
            result_t1 = self.tracker(
                get_input_sample(frame_index=i),
                detections,
                embeddings=embeddings,
            ).boxes2d[0]

        self.assertTrue(
            (torch.sum(generated_detections[0].score > self.min_confidence))
            == len(result_t1)
        )
