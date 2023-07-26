# pylint: disable=unexpected-keyword-arg
"""CC-3DT model test file."""
import unittest

import torch

from tests.util import get_test_data, get_test_file
from vis4d.common.ckpt import load_model_checkpoint
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets.nuscenes import NuScenes, nuscenes_class_map
from vis4d.data.loader import build_inference_dataloaders, multi_sensor_collate
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.resize import (
    GenResizeParameters,
    ResizeImages,
    ResizeIntrinsics,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.engine.connectors import MultiSensorDataConnector
from vis4d.model.track3d.cc_3dt import FasterRCNNCC3DT, Track3DOut
from vis4d.zoo.cc_3dt.data import CONN_NUSC_BBOX_3D_TEST


class CC3DTTest(unittest.TestCase):
    """CC-3DT class tests."""

    model_weights_prefix = "https://dl.cv.ethz.ch/vis4d/cc_3dt/"

    def test_r50_fpn_inference(self):
        """Inference test."""
        model_weights = (
            self.model_weights_prefix
            + "cc_3dt_frcnn_r50_fpn_12e_nusc_d98509.pt"
        )

        cc_3dt = FasterRCNNCC3DT(num_classes=len(nuscenes_class_map))

        load_model_checkpoint(cc_3dt, model_weights, strict=True)

        preprocess_fn = compose(
            [
                GenResizeParameters(
                    shape=(256, 704), keep_ratio=True, sensors=NuScenes.CAMERAS
                ),
                ResizeImages(sensors=NuScenes.CAMERAS),
                ResizeIntrinsics(sensors=NuScenes.CAMERAS),
            ]
        )

        batch_fn = compose(
            [
                PadImages(sensors=NuScenes.CAMERAS),
                NormalizeImages(sensors=NuScenes.CAMERAS),
                ToTensor(sensors=NuScenes.CAMERAS),
            ]
        )

        dataset = NuScenes(
            data_root=get_test_data("nuscenes_test"),
            keys_to_load=[K.images, K.original_images, K.boxes3d],
            version="v1.0-mini",
            split="mini_val",
        )
        datapipe = DataPipe(dataset, preprocess_fn=preprocess_fn)
        test_loader = build_inference_dataloaders(
            datapipe,
            samples_per_gpu=1,
            workers_per_gpu=1,
            batchprocess_fn=batch_fn,
            collate_fn=multi_sensor_collate,
            sensors=NuScenes.CAMERAS,
        )[0]

        data_connector = MultiSensorDataConnector(
            key_mapping=CONN_NUSC_BBOX_3D_TEST
        )

        cc_3dt.eval()

        tracks_list = []
        with torch.no_grad():
            for cur_iter, data in enumerate(test_loader):
                test_input = data_connector(data)

                tracks = cc_3dt(**test_input)
                assert isinstance(tracks, Track3DOut)

                tracks_list.append(tracks)

                if cur_iter == 1:
                    break

        testcase_gt_list = torch.load(get_test_file("cc_3dt.pt"))
        for tracks, testcase_gt in zip(tracks_list, testcase_gt_list):
            for pred, expected in zip(tracks, testcase_gt):
                for pred_entry, expected_entry in zip(pred, expected):
                    assert (
                        torch.isclose(pred_entry, expected_entry, atol=1e-2)
                        .all()
                        .item()
                    )
