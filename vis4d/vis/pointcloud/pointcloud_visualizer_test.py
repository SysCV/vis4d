"""TODO."""

import glob
import shutil
import tempfile
import unittest

import numpy as np
import open3d as o3d
import torch

from vis4d.data.const import CommonKeys
from vis4d.data.datasets.s3dis import S3DIS
from vis4d.data.loader import DataPipe, SubdividingIterableDataset
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.point_sampling import (
    sample_points_block_full_coverage,
)
from vis4d.unittest.util import get_test_data, get_test_file
from vis4d.vis.pointcloud.pointcloud_visualizer import PointCloudVisualizer


class TestPointcloudViewer(unittest.TestCase):
    """Test Pointcloud viewer."""

    def setUp(self) -> None:
        """Creates a tmp directory and loads input data."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Removes the tmp directory."""
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def _assert_pc_equal(self, file1: str, file2: str) -> None:
        """Checks that the pointcloud stored at the given two paths are equal

        Args:
            file1: Path to pc1
            file2:  Path to pc2
        """
        pc1 = o3d.io.read_point_cloud(file1)
        pc2 = o3d.io.read_point_cloud(file2)

        self.assertTrue(
            np.allclose(np.asarray(pc1.points), np.asarray(pc2.points))
        )
        self.assertTrue(
            np.allclose(np.asarray(pc1.colors), np.asarray(pc2.colors))
        )

    def test_precomputed(self) -> None:
        """Loads a precomputed datasamples from s3dis and checks the output."""
        test_file_loc = get_test_file("test_s3dis_pts_in.pt")
        data = torch.load(test_file_loc)
        vis = PointCloudVisualizer()
        for e in data:
            vis.process(
                points_xyz=e[CommonKeys.points3d].numpy(),
                semantics=e[CommonKeys.semantics3d].numpy(),
                colors=e[CommonKeys.colors3d].numpy(),
                instances=e[CommonKeys.instances3d].numpy(),
                scene_index=e["source_index"].numpy(),
            )

        vis.save_to_disk(self.test_dir)
        for f in glob.glob(self.test_dir + "/**/*.ply"):
            self._assert_pc_equal(
                f,
                f.replace(
                    self.test_dir, get_test_data("pointcloud_vis/s3dis")
                ),
            )

    def test_vis_s3dis(self) -> None:
        """Loads two rooms from the s3dis dataset and visualizes it."""
        ds = S3DIS(data_root=get_test_data("s3d_test"))
        sample = sample_points_block_full_coverage(  # pylint: disable=unexpected-keyword-arg,line-too-long
            in_keys=[
                CommonKeys.points3d,
                CommonKeys.colors3d,
                CommonKeys.semantics3d,
                CommonKeys.instances3d,
            ],
            out_keys=[
                CommonKeys.points3d,
                CommonKeys.colors3d,
                CommonKeys.semantics3d,
                CommonKeys.instances3d,
            ],
            n_pts_per_block=4096,
            min_pts_per_block=512,
        )

        datapipe = SubdividingIterableDataset(
            DataPipe(ds, compose([sample])),
            n_samples_per_batch=4096,
            preprocess_fn=lambda x: x,
        )

        vis = PointCloudVisualizer()
        for e in datapipe:
            vis.process(
                points_xyz=e[CommonKeys.points3d].numpy(),
                semantics=e[CommonKeys.semantics3d].numpy(),
                colors=e[CommonKeys.colors3d].numpy(),
                instances=e[CommonKeys.instances3d].numpy(),
                scene_index=e["source_index"].numpy(),
            )

        vis.save_to_disk(self.test_dir)
