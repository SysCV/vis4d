"""CC-3DT model test file."""
import os.path as osp
import unittest

import torch

from tests.util import get_test_data
from vis4d.data.const import CommonKeys
from vis4d.data.datasets.nuscenes import NuScenes
from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    multi_sensor_collate,
)
from vis4d.data.transforms import compose
from vis4d.data.transforms.normalize import batched_normalize_image
from vis4d.data.transforms.pad import pad_image
from vis4d.data.transforms.resize import resize_image, resize_intrinsics
from vis4d.engine.ckpt import load_model_checkpoint
from vis4d.model.track3d.cc_3dt import FasterRCNNCC3DT


class CC3DTTest(unittest.TestCase):
    """CC-3DT class tests."""

    # TODO: finish test
    model_weights = "https://dl.cv.ethz.ch/vis4d/cc_3dt_R_50_FPN_nuscenes_12_accumulate_gradient_2.ckpt"

    def test_inference(self):
        """Inference test.

        Run::
            >>> pytest tests/model/track3d/cc_3dt_test.py::CC3DTTest::test_inference
        """
        cc_3dt = FasterRCNNCC3DT(num_classes=10)
        load_model_checkpoint(cc_3dt, self.model_weights)

        data_root = osp.join(get_test_data("nuscenes_test"), "track3d/images")
        preprocess_fn = compose(
            [
                resize_image(
                    shape=(900, 1600),
                    keep_ratio=True,
                    sensors=NuScenes._CAMERAS,
                ),
                resize_intrinsics(sensors=NuScenes._CAMERAS),
            ]
        )

        test_data = DataPipe(
            NuScenes(
                data_root,  # TODO: check nuscenes test data
                version="v1.0-mini",
                split="mini_val",
                metadata=["use_camera"],
            ),
            preprocess_fn=preprocess_fn,
        )
        batch_fn = compose(
            [
                pad_image(sensors=NuScenes._CAMERAS),
                batched_normalize_image(sensors=NuScenes._CAMERAS),
            ]
        )
        batch_size = 1
        test_loader = build_inference_dataloaders(
            test_data,
            samples_per_gpu=batch_size,
            workers_per_gpu=0,
            batchprocess_fn=batch_fn,
            collate_fn=multi_sensor_collate,
        )[0]

        data = next(iter(test_loader))
        # assume: inputs are consecutive frames
        images = []
        inputs_hw = []
        frame_ids = []
        intrinscs = []
        extrinsics = []
        for cam in NuScenes._CAMERAS:
            images.append(data[cam][CommonKeys.images])
            inputs_hw.extend(data[cam][CommonKeys.original_hw])
            intrinscs.append(data[cam][CommonKeys.intrinsics])
            extrinsics.append(data[cam][CommonKeys.extrinsics])
            frame_ids.append(data[cam][CommonKeys.frame_ids])

        images = torch.cat(images, dim=0)
        intrinsics = torch.cat(intrinscs, dim=0)
        extrinsics = torch.cat(extrinsics, dim=0)

        cc_3dt.eval()
        with torch.no_grad():
            tracks = cc_3dt(  # pylint: disable=unused-variable
                images, inputs_hw, intrinsics, extrinsics, frame_ids
            )
        # FIXME
        # testcase_gt = torch.load(get_test_file("cc_3dt.pt"))
        # for pred, expected in zip(tracks, testcase_gt):
        #     for pred_entry, expected_entry in zip(pred, expected):
        #         pass
        #         assert (
        #             torch.isclose(pred_entry, expected_entry, atol=1e-4)
        #             .all()
        #             .item()
        #         )

    # def test_train(self): #FIXME
    #     """Training test."""
    #     pass

    # def test_torchscript(self):  #FIXME
    #     """Test torchscipt export."""
    #     sample_images = torch.rand((2, 3, 512, 512))
    #     qdtrack = FasterRCNNQDTrack(num_classes=8)
    #     qdtrack_scripted = torch.jit.script(qdtrack)
    #     qdtrack_scripted(sample_images)
