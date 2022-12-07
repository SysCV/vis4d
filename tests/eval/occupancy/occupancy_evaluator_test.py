"""Track visualziation test cases."""
from __future__ import annotations

import unittest

import numpy as np

from vis4d.common.typing import NDArrayNumber
from vis4d.eval.occupancy.occupancy_evaluator import OccupancyEvaluator


def get_test_data() -> tuple[NDArrayNumber, NDArrayNumber]:
    """Precomputed test data."""
    prediction = np.asarray(
        [
            [
                0.74237575,
                0.15012051,
                0.12699674,
                0.90514564,
                0.88827831,
                0.16842676,
                0.05836188,
                0.42198188,
                0.86834491,
                0.8819624,
                0.12740623,
                0.04748418,
                0.12094138,
                0.60813396,
                0.04572285,
                0.61252886,
                0.79309815,
                0.18766109,
                0.61808356,
                0.42195592,
                0.67309237,
                0.79049787,
                0.35847636,
                0.72546534,
                0.63627838,
            ],
            [
                0.48552076,
                0.13560225,
                0.79410182,
                0.05379457,
                0.62496323,
                0.86515683,
                0.80848697,
                0.230458,
                0.75289785,
                0.72078345,
                0.53457811,
                0.48931605,
                0.62778233,
                0.09268421,
                0.35686759,
                0.61617648,
                0.09824877,
                0.1197167,
                0.28514205,
                0.79445318,
                0.06004223,
                0.02166305,
                0.80442794,
                0.81305464,
                0.74295228,
            ],
            [
                0.71079104,
                0.41384821,
                0.39415346,
                0.96155834,
                0.42142236,
                0.71920926,
                0.73248142,
                0.58856593,
                0.37915575,
                0.71009794,
                0.04974543,
                0.66322415,
                0.21537487,
                0.80972038,
                0.54723062,
                0.53931716,
                0.09919259,
                0.34545742,
                0.39561214,
                0.53715127,
                0.97846143,
                0.88463563,
                0.9556771,
                0.42538701,
                0.70962377,
            ],
            [
                0.16960425,
                0.11385814,
                0.56957041,
                0.9155517,
                0.58739935,
                0.6130923,
                0.5719799,
                0.65234523,
                0.27342812,
                0.94933045,
                0.30975219,
                0.30584413,
                0.08499566,
                0.30885289,
                0.81021219,
                0.40411665,
                0.74205046,
                0.64253772,
                0.38613562,
                0.03185519,
                0.91610862,
                0.08285329,
                0.19013149,
                0.60908678,
                0.27017079,
            ],
        ]
    )

    gt = np.asarray(
        [
            [
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                True,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
            [
                True,
                True,
                True,
                True,
                False,
                True,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
            ],
            [
                True,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
            ],
            [
                True,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                False,
            ],
        ]
    )
    return prediction, gt


class TestOccupancyEvaluator(unittest.TestCase):
    """Tests for OccupancyEvaluator."""

    evaluator = OccupancyEvaluator()
    batch_size = 4
    n_points = 100

    def test_confusion_all_ones(self) -> None:
        """Tests when gt is all ones."""
        # All ones
        pred = np.ones((self.batch_size, self.n_points)) * 0.6
        gt = np.ones((self.batch_size, self.n_points))
        self.evaluator.reset()
        self.evaluator.process(pred, gt)
        data, _ = self.evaluator.evaluate(OccupancyEvaluator.METRIC_ALL)
        self.assertAlmostEqual(data[OccupancyEvaluator.METRIC_RECALL], 1)
        self.assertAlmostEqual(data[OccupancyEvaluator.METRIC_ACCURACY], 1)
        self.assertAlmostEqual(data[OccupancyEvaluator.METRIC_PRECISION], 1)

    def test_confusion_all_ones_threshold_08(self) -> None:
        """Tests prediction all ones with different threshold."""
        # All ones
        evaluator = OccupancyEvaluator(threshold=0.3)
        pred = np.ones((self.batch_size, self.n_points)) * 0.34
        gt = np.ones((self.batch_size, self.n_points))
        evaluator.process(pred, gt)
        data, _ = evaluator.evaluate(OccupancyEvaluator.METRIC_ALL)
        self.assertAlmostEqual(data[OccupancyEvaluator.METRIC_RECALL], 1)
        self.assertAlmostEqual(data[OccupancyEvaluator.METRIC_ACCURACY], 1)
        self.assertAlmostEqual(data[OccupancyEvaluator.METRIC_PRECISION], 1)

    def test_confusion_all_zeros(self) -> None:
        """Tests when data is all zeros."""
        # All ones
        pred = np.ones((self.batch_size, self.n_points)) * 0.2
        gt = np.zeros((self.batch_size, self.n_points))
        self.evaluator.reset()
        self.evaluator.process(pred, gt)
        data, _ = self.evaluator.evaluate(OccupancyEvaluator.METRIC_ALL)
        self.assertAlmostEqual(data[OccupancyEvaluator.METRIC_RECALL], 1)
        self.assertAlmostEqual(data[OccupancyEvaluator.METRIC_ACCURACY], 1)
        self.assertAlmostEqual(data[OccupancyEvaluator.METRIC_PRECISION], 1)

    def test_precomputed(self) -> None:
        """Tests values for precomputed data."""
        prediction, gt = get_test_data()
        self.evaluator.reset()
        self.evaluator.process(prediction, gt)
        data, _ = self.evaluator.evaluate(OccupancyEvaluator.METRIC_ALL)
        self.assertAlmostEqual(data[OccupancyEvaluator.METRIC_RECALL], 0.47500)
        self.assertAlmostEqual(
            data[OccupancyEvaluator.METRIC_ACCURACY], 0.4500
        )
        self.assertAlmostEqual(
            data[OccupancyEvaluator.METRIC_PRECISION], 0.3584905660377358
        )
        self.assertAlmostEqual(
            data[OccupancyEvaluator.METRIC_F1], 0.4086021505376344
        )
        self.assertAlmostEqual(
            data[OccupancyEvaluator.METRIC_IOU], 0.2567567567567
        )

    def test_batched_precomputed(self) -> None:
        """Tests precomputed values when provided over two batches."""
        prediction, gt = get_test_data()
        n_batch = prediction.shape[0] // 2
        self.evaluator.reset()
        self.evaluator.process(prediction[:n_batch, ...], gt[:n_batch, ...])
        self.evaluator.process(prediction[-n_batch:, ...], gt[-n_batch:, ...])

        data, _ = self.evaluator.evaluate(OccupancyEvaluator.METRIC_ALL)

        self.assertAlmostEqual(data[OccupancyEvaluator.METRIC_RECALL], 0.47500)
        self.assertAlmostEqual(
            data[OccupancyEvaluator.METRIC_ACCURACY], 0.4500
        )
        self.assertAlmostEqual(
            data[OccupancyEvaluator.METRIC_PRECISION], 0.3584905660377358
        )
        self.assertAlmostEqual(
            data[OccupancyEvaluator.METRIC_F1], 0.4086021505376344
        )
        self.assertAlmostEqual(
            data[OccupancyEvaluator.METRIC_IOU], 0.2567567567567
        )
