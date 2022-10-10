"""point sampling augmentation testing class."""
import copy
import unittest

import pytest
import torch

from vis4d.common import COMMON_KEYS

from .points import move_pts_to_last_channel


class TestPoints(unittest.TestCase):
    """Tests sampling in a block based fashion."""

    @pytest.fixture(autouse=True)
    def initdata(self):
        """Loads dummy data."""
        self.data = {
            COMMON_KEYS.points3d: torch.rand(200, 3),
            COMMON_KEYS.colors3d: torch.rand(200, 3),
            COMMON_KEYS.semantics3d: torch.randint(10, (200, 1)),
        }
        self.original_data = copy.deepcopy(self.data)

    def test_move_pts_to_last_channel(self):
        """Tests the functional."""
        tf = move_pts_to_last_channel(
            in_keys=[COMMON_KEYS.points3d], out_keys=[COMMON_KEYS.points3d]
        )

        # Check that points are now at last channel
        self.assertEqual(tf(self.data)[COMMON_KEYS.points3d].shape[-1], 200)

    def test_move_pts_to_last_channel_w_multi_keys(self):
        """TODO"""
        # Check mutli key case
        tf = move_pts_to_last_channel(
            in_keys=[
                COMMON_KEYS.points3d,
                COMMON_KEYS.colors3d,
                COMMON_KEYS.semantics3d,
            ],
            out_keys=[
                COMMON_KEYS.points3d,
                COMMON_KEYS.colors3d,
                COMMON_KEYS.semantics3d,
            ],
        )

        # Check that num_points are now at last channel
        out = tf(self.data)
        for data in out.values():
            self.assertEqual(data.shape[-1], 200)
