"""Tests for the 3D bounding box visualizer."""
import os
import shutil
import tempfile
import unittest

import torch

from tests.util import get_test_data, get_test_file
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.nuscenes import NuScenes, nuscenes_class_map
from vis4d.data.loader import build_inference_dataloaders, multi_sensor_collate
from vis4d.engine.connectors import MultiSensorCallbackConnector
from vis4d.model.track3d.cc_3dt import Track3DOut
from vis4d.vis.image.bbox3d_visualizer import (
    BoundingBox3DVisualizer,
    MultiCameraBBox3DVisualizer,
)
from vis4d.zoo.cc_3dt.data import CONN_NUSC_BBOX_3D_VIS

from .util import compare_images


class TestBoundingBox3DVis(unittest.TestCase):
    """Testcase for Bounding Box 3D Visualizer."""

    def setUp(self) -> None:
        """Creates a tmp directory and loads input data."""
        self.test_dir = tempfile.mkdtemp()
        data_root = get_test_data("nuscenes_test", absolute_path=False)

        self.testcase_gt = torch.load(f"{data_root}/cc_3dt.pt")

        dataset = NuScenes(
            data_root=data_root,
            keys_to_load=[K.images, K.original_images, K.boxes3d],
            version="v1.0-mini",
            split="mini_val",
            cache_as_binary=True,
            cached_file_path=f"{data_root}/mini_val.pkl",
        )

        self.test_loader = build_inference_dataloaders(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=1,
            collate_fn=multi_sensor_collate,
            sensors=NuScenes.CAMERAS,
        )[0]

    def tearDown(self) -> None:
        """Removes the tmp directory after the test."""
        shutil.rmtree(self.test_dir)

    def test_single_camera(self) -> None:
        """Test visualization of 3D bboxes with single camera."""
        vis = BoundingBox3DVisualizer(
            cat_mapping=nuscenes_class_map, vis_freq=1
        )

        for cur_iter, data in enumerate(self.test_loader):
            track_result: Track3DOut = self.testcase_gt[cur_iter]

            image_names = data["CAM_FRONT"][K.sample_names]
            sequence_names = data[K.sequence_names]

            vis.process(
                cur_iter=cur_iter,
                images=data["CAM_FRONT"][K.images],
                image_names=image_names,
                boxes3d=track_result.boxes_3d,
                intrinsics=data["CAM_FRONT"][K.intrinsics],
                extrinsics=data["CAM_FRONT"][K.extrinsics],
                scores=track_result.scores_3d,
                class_ids=track_result.class_ids,
                track_ids=track_result.track_ids,
                sequence_names=data[K.sequence_names],
            )

            vis.save_to_disk(cur_iter=cur_iter, output_folder=self.test_dir)

            for batch, sequence_name in enumerate(sequence_names):
                self.assertTrue(
                    compare_images(
                        os.path.join(
                            self.test_dir,
                            f"{sequence_name}/{image_names[batch]}.png",
                        ),
                        get_test_file(f"bbox3d/{image_names[batch]}.png"),
                    )
                )

            vis.reset()

            if cur_iter == 1:
                break

    def test_multi_camera(self) -> None:
        """Test visualization of 3D bboxes with multi cameras."""
        vis = MultiCameraBBox3DVisualizer(
            cat_mapping=nuscenes_class_map,
            vis_freq=1,
            cameras=NuScenes.CAMERAS,
        )

        data_connector = MultiSensorCallbackConnector(
            key_mapping=CONN_NUSC_BBOX_3D_VIS
        )

        for cur_iter, data in enumerate(self.test_loader):
            track_result: Track3DOut = self.testcase_gt[cur_iter]

            visualizer_inputs = data_connector(track_result, data)

            sequence_names = data[K.sequence_names]

            vis.process(cur_iter=cur_iter, **visualizer_inputs)

            vis.save_to_disk(cur_iter=cur_iter, output_folder=self.test_dir)

            test_cameras = ["CAM_FRONT", "CAM_BACK_RIGHT"]

            for camera in test_cameras:
                image_names = data[camera][K.sample_names]
                for batch, sequence_name in enumerate(sequence_names):
                    self.assertTrue(
                        compare_images(
                            os.path.join(
                                self.test_dir,
                                sequence_name,
                                camera,
                                f"{image_names[batch]}.png",
                            ),
                            get_test_file(
                                f"bbox3d/{camera}/{image_names[batch]}.png"
                            ),
                        )
                    )

            vis.reset()

            if cur_iter == 1:
                break
