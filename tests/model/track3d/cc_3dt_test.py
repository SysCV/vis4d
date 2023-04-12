# pylint: disable=unexpected-keyword-arg
"""CC-3DT model test file."""
import unittest

import torch

from tests.util import get_test_data, get_test_file
from vis4d.data.const import CommonKeys as CK
from vis4d.data.datasets.nuscenes import (
    NuScenes,
    nuscenes_class_range_map,
    nuscenes_track_map,
)
from vis4d.data.loader import (
    VideoDataPipe,
    build_inference_dataloaders,
    multi_sensor_collate,
)
from vis4d.data.transforms.base import compose, compose_batch
from vis4d.data.transforms.normalize import BatchNormalizeImages
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.resize import (
    GenerateResizeParameters,
    ResizeImage,
    ResizeIntrinsics,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.engine.connectors import (
    DataConnectionInfo,
    MultiSensorDataConnector,
)
from vis4d.model.track3d.cc_3dt import FasterRCNNCC3DT, Track3DOut


class CC3DTTest(unittest.TestCase):  # TODO: add training test
    """CC-3DT class tests."""

    model_weights = "https://dl.cv.ethz.ch/vis4d/cc_3dt_R_50_FPN_nuscenes.pt"

    CONN_BBOX_3D_TEST = {
        CK.images: CK.images,
        CK.original_hw: "images_hw",
        CK.intrinsics: CK.intrinsics,
        CK.extrinsics: CK.extrinsics,
        CK.frame_ids: CK.frame_ids,
    }

    CAMERAS = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def test_inference(self):
        """Inference test."""
        cc_3dt = FasterRCNNCC3DT(
            num_classes=len(nuscenes_track_map),
            class_range_map=torch.Tensor(nuscenes_class_range_map),
            weights=self.model_weights,
        )

        preprocess_fn = compose(
            [
                GenerateResizeParameters(
                    shape=(900, 1600),
                    keep_ratio=True,
                    sensors=self.CAMERAS,
                ),
                ResizeImage(sensors=self.CAMERAS),
                ResizeIntrinsics(sensors=self.CAMERAS),
            ]
        )

        batch_fn = compose_batch(
            [
                PadImages(sensors=self.CAMERAS),
                BatchNormalizeImages(sensors=self.CAMERAS),
                ToTensor(sensors=self.CAMERAS),
            ]
        )

        dataset = NuScenes(
            data_root=get_test_data("nuscenes_test"),
            version="v1.0-mini",
            split="mini_val",
            metadata=["use_camera"],
        )
        datapipe = VideoDataPipe(
            dataset,
            preprocess_fn=preprocess_fn,
        )
        test_loader = build_inference_dataloaders(
            datapipe,
            samples_per_gpu=1,
            workers_per_gpu=1,
            batchprocess_fn=batch_fn,
            collate_fn=multi_sensor_collate,
        )[0]

        data_connection_info = DataConnectionInfo(
            test=self.CONN_BBOX_3D_TEST,
        )

        data_connector = MultiSensorDataConnector(
            connections=data_connection_info,
            default_sensor="CAM_FRONT",
            sensors=self.CAMERAS,
        )

        cc_3dt.eval()

        tracks_list = []
        with torch.no_grad():
            for cur_iter, data in enumerate(test_loader):
                test_input = data_connector.get_test_input(data)

                tracks = cc_3dt(**test_input)
                assert isinstance(tracks, Track3DOut)

                tracks_list.append(tracks)

                if cur_iter == 1:
                    break

        # TODO: The current results don't use bev nms due to detetron2
        # dependency. We will need to update the test case.
        testcase_gt_list = torch.load(get_test_file("cc_3dt.pt"))
        for tracks, testcase_gt in zip(tracks_list, testcase_gt_list):
            for pred, expected in zip(tracks, testcase_gt):
                for pred_entry, expected_entry in zip(pred, expected):
                    assert (
                        torch.isclose(pred_entry, expected_entry, atol=1)
                        .all()
                        .item()
                    )
