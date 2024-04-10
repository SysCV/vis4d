"""Testcases for reference view sampling."""

import unittest

from vis4d.data.reference import SequentialViewSampler, UniformViewSampler


class UniformViewSamplerTest(unittest.TestCase):
    """Testcases for uniform view sampling."""

    key_index = 42

    indices_in_video = [30, 36, 42, 48, 54, 60, 66, 72]
    frame_ids = [30, 31, 32, 33, 34, 35, 36, 37]

    def test_2_scope_1_sample(self) -> None:
        """Test uniform sampler with scope of 2 and 1 reference samples."""
        sampler = UniformViewSampler(scope=2, num_ref_samples=1)

        ref_indices = sampler(
            self.key_index, self.indices_in_video, self.frame_ids
        )
        assert ref_indices[0] in {30, 36, 48, 54}

        # test non consecutive frame ids
        indices_in_video = [30, 42, 54, 60, 66, 72]
        frame_ids = [30, 32, 34, 35, 36, 37]

        ref_indices = sampler(self.key_index, indices_in_video, frame_ids)
        assert ref_indices[0] in {30, 54}

        indices_in_video = [30, 36, 42, 60, 66, 72]
        frame_ids = [30, 31, 32, 35, 36, 37]

        ref_indices = sampler(self.key_index, indices_in_video, frame_ids)
        assert ref_indices[0] in {30, 36}

        # test non valid indices
        indices_in_video = [42, 60, 66, 72]
        frame_ids = [32, 35, 36, 37]

        ref_indices = sampler(self.key_index, indices_in_video, frame_ids)
        assert ref_indices[0] == 42

    def test_1_scope_2_sample(self) -> None:
        """Test uniform sampler with scope of 1 and 2 reference samples."""
        sampler = UniformViewSampler(scope=1, num_ref_samples=2)

        # video indices
        ref_indices = sampler(
            self.key_index, self.indices_in_video, self.frame_ids
        )
        assert set(ref_indices) == {36, 48}

        # test non valid indices
        indices_in_video = [42, 60, 66, 72]
        frame_ids = [32, 35, 36, 37]

        ref_indices = sampler(self.key_index, indices_in_video, frame_ids)
        assert ref_indices == [42, 42]


class SequentialViewSamplerTest(unittest.TestCase):
    """Testcases for sequential view sampling."""

    key_index = 42

    indices_in_video = [30, 36, 42, 48, 54, 60, 66, 72]
    frame_ids = [30, 31, 32, 33, 34, 35, 36, 37]

    def test_1_sample(self) -> None:
        """Test uniform sampler with scope of 2 and 1 reference samples."""
        sampler = SequentialViewSampler(num_ref_samples=1)

        ref_indices = sampler(
            self.key_index, self.indices_in_video, self.frame_ids
        )
        assert ref_indices[0] == 48

        # test non consecutive frame ids
        indices_in_video = [30, 42, 54, 60, 66, 72]
        frame_ids = [30, 32, 34, 35, 36, 37]

        ref_indices = sampler(self.key_index, indices_in_video, frame_ids)
        assert ref_indices[0] == 54

        indices_in_video = [30, 36, 42, 60, 66, 72]
        frame_ids = [30, 31, 32, 35, 36, 37]

        ref_indices = sampler(self.key_index, indices_in_video, frame_ids)
        assert ref_indices[0] == 60
