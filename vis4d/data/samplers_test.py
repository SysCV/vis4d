"""Testcases for Vis4D data samplers."""
import unittest

from torch.utils.data import ConcatDataset, Dataset

from .samplers import RoundRobinSampler


class MockDataset(Dataset):  # type: ignore
    """PyTorch dataset mockup."""

    def __init__(self, start: int, end: int) -> None:
        """Init."""
        super().__init__()
        assert end > start
        self.start, self.end = start, end

    def __len__(self) -> int:
        """Len."""
        return self.end - self.start

    def __getitem__(self, item: int) -> None:
        """Getitem."""


class TestRoundRobinSampler(unittest.TestCase):
    """Round-robin sampler testcase class."""

    dataset = ConcatDataset(
        [MockDataset(0, 5), MockDataset(5, 10), MockDataset(10, 15)]
    )
    unbal_dset = ConcatDataset(
        [
            MockDataset(0, 5),
            MockDataset(5, 7),
            MockDataset(7, 8),
            MockDataset(8, 10),
            MockDataset(10, 11),
        ]
    )

    def test_single_sampling(self) -> None:
        """Testcase for batch=1 sampling."""
        sampler = RoundRobinSampler(
            self.dataset,
            1,
            repeat_interval=0,
            spread_samples=True,
            max_samples=-1,
        )
        self.assertEqual(len(sampler), len(self.dataset))
        samp_it = iter(sampler)
        for i in range(len(sampler)):
            batch = next(samp_it)
            self.assertTrue(isinstance(batch, list))
            self.assertEqual(len(batch), 1)
            self.assertTrue(5 * i % 3 <= batch[0] < 5 * (i % 3 + 1))

    def test_multi_sampling(self) -> None:
        """Testcase for batch>1 sampling."""
        sampler = RoundRobinSampler(
            self.dataset,
            2,
            repeat_interval=0,
            spread_samples=True,
            max_samples=-1,
        )
        samp_it = iter(sampler)
        for i in range(len(sampler)):
            batch = next(samp_it)
            self.assertTrue(isinstance(batch, list))
            self.assertLessEqual(len(batch), 2)
            for ind in batch:
                self.assertTrue(5 * i % 3 <= ind < 5 * (i % 3 + 1))

    def test_repeat(self) -> None:
        """Testcase for repeat sampling."""
        sampler = RoundRobinSampler(
            self.unbal_dset,
            1,
            repeat_interval=[0, 1, 2, 0, 0],
            spread_samples=[True, True, False, True, False],
            max_samples=[4, -1, -1, -1, -1],
        )
        self.assertEqual(len(sampler), 13)
        samp_it = iter(sampler)

        batch = next(samp_it)
        self.assertTrue(0 <= batch[0] < 5)
        batch = next(samp_it)
        self.assertTrue(5 <= batch[0] < 7)
        batch = next(samp_it)
        self.assertTrue(7 <= batch[0] < 8)
        batch = next(samp_it)
        self.assertTrue(8 <= batch[0] < 10)
        batch = next(samp_it)
        self.assertTrue(10 <= batch[0] < 11)

        batch = next(samp_it)
        self.assertTrue(0 <= batch[0] < 5)
        batch = next(samp_it)
        self.assertTrue(5 <= batch[0] < 7)

        batch = next(samp_it)
        self.assertTrue(0 <= batch[0] < 5)
        batch = next(samp_it)
        self.assertTrue(5 <= batch[0] < 7)
        batch = next(samp_it)
        self.assertTrue(7 <= batch[0] < 8)
        batch = next(samp_it)
        self.assertTrue(8 <= batch[0] < 10)

        batch = next(samp_it)
        self.assertTrue(0 <= batch[0] < 5)
        batch = next(samp_it)
        self.assertTrue(5 <= batch[0] < 7)
