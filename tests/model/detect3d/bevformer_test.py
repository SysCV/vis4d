# pylint: disable=unexpected-keyword-arg
"""CC-3DT model test file."""
import unittest

import torch

from tests.util import get_test_data, get_test_file
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets.nuscenes import NuScenes
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
from vis4d.model.detect3d.bevformer import BEVFormer
from vis4d.op.base import ResNet
from vis4d.op.detect3d.bevformer import BEVFormerHead
from vis4d.op.detect3d.bevformer.encoder import (
    BEVFormerEncoder,
    BEVFormerEncoderLayer,
)
from vis4d.op.detect3d.bevformer.spatial_cross_attention import (
    MSDeformableAttention3D,
    SpatialCrossAttention,
)
from vis4d.op.detect3d.bevformer.transformer import PerceptionTransformer
from vis4d.op.detect3d.common import Detect3DOut
from vis4d.op.fpp.fpn import FPN
from vis4d.zoo.bevformer.data import (
    CONN_NUSC_BBOX_3D_TEST,
    NUSC_CAMERAS,
    NUSC_SENSORS,
)


class CC3DTTest(unittest.TestCase):
    """CC-3DT class tests."""

    model_weights_prefix = (
        "https://github.com/zhiqi-li/storage/releases/download/v1.0/"
    )

    def test_tiny_inference(self):
        """Inference test."""
        data_root = get_test_data("nuscenes_test", absolute_path=False)
        model_weights = (
            self.model_weights_prefix + "bevformer_tiny_epoch_24.pth"
        )

        bevformer_tiny = BEVFormer(
            basemodel=ResNet(resnet_name="resnet50", trainable_layers=3),
            fpn=FPN(
                in_channels_list=[2048],
                out_channels=256,
                extra_blocks=None,
                start_index=5,
            ),
            pts_bbox_head=BEVFormerHead(
                transformer=PerceptionTransformer(
                    encoder=BEVFormerEncoder(
                        layer=BEVFormerEncoderLayer(
                            cross_attn=SpatialCrossAttention(
                                deformable_attention=MSDeformableAttention3D(
                                    num_levels=1,
                                ),
                            ),
                        ),
                        num_layers=3,
                    ),
                ),
                bev_h=50,
                bev_w=50,
            ),
            weights=model_weights,
        )

        preprocess_fn = compose(
            [
                GenResizeParameters(
                    shape=(450, 800), keep_ratio=True, sensors=NUSC_CAMERAS
                ),
                ResizeImages(sensors=NUSC_CAMERAS),
                ResizeIntrinsics(sensors=NUSC_CAMERAS),
                NormalizeImages(sensors=NUSC_CAMERAS),
            ]
        )

        batch_fn = compose(
            [PadImages(sensors=NUSC_CAMERAS), ToTensor(sensors=NUSC_SENSORS)]
        )

        dataset = NuScenes(
            data_root=data_root,
            keys_to_load=[K.images, K.original_images, K.boxes3d],
            version="v1.0-mini",
            split="mini_val",
            cache_as_binary=True,
            cached_file_path=f"{data_root}/mini_val.pkl",
        )
        datapipe = DataPipe(dataset, preprocess_fn=preprocess_fn)
        test_loader = build_inference_dataloaders(
            datapipe,
            samples_per_gpu=1,
            workers_per_gpu=1,
            video_based_inference=True,
            batchprocess_fn=batch_fn,
            collate_fn=multi_sensor_collate,
            sensors=NUSC_SENSORS,
        )[0]

        data_connector = MultiSensorDataConnector(
            key_mapping=CONN_NUSC_BBOX_3D_TEST
        )

        bevformer_tiny.eval()

        dets_list = []
        with torch.no_grad():
            for cur_iter, data in enumerate(test_loader):
                test_input = data_connector(data)

                dets = bevformer_tiny(**test_input)
                assert isinstance(dets, Detect3DOut)

                dets_list.append(dets)

                if cur_iter == 1:
                    break

        testcase_gt_list = torch.load(
            get_test_file("bevformer_tiny.pt"), weights_only=False
        )

        for dets, testcase_gt in zip(dets_list, testcase_gt_list):
            for pred, expected in zip(dets, testcase_gt):
                for pred_entry, expected_entry in zip(pred, expected):
                    assert (
                        torch.isclose(pred_entry, expected_entry, atol=1e-1)
                        .all()
                        .item()
                    )
