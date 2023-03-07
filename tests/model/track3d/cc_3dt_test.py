"""CC-3DT model test file."""
# import os.path as osp
# import unittest

# import torch

# from tests.util import get_test_data
# from vis4d.data.const import CommonKeys
# from vis4d.data.datasets.nuscenes import (
#     NuScenes,
#     nuscenes_class_range_map,
#     nuscenes_track_map,
# )
# from vis4d.data.loader import (
#     VideoDataPipe,
#     build_inference_dataloaders,
#     multi_sensor_collate,
# )
# from vis4d.data.transforms import compose
# from vis4d.data.transforms.normalize import batched_normalize_image
# from vis4d.data.transforms.pad import pad_image
# from vis4d.data.transforms.resize import resize_image, resize_intrinsics
# from vis4d.model.track3d.cc_3dt import FasterRCNNCC3DT


# class CC3DTTest(unittest.TestCase):
#     """CC-3DT class tests."""

#     # TODO: finish test
#     model_weights = "https://dl.cv.ethz.ch/vis4d/cc_3dt_R_50_FPN_nuscenes.pt"

#     cameras = [
#         "LIDAR_TOP",
#         "CAM_FRONT",
#         "CAM_FRONT_LEFT",
#         "CAM_FRONT_RIGHT",
#         "CAM_BACK",
#         "CAM_BACK_LEFT",
#         "CAM_BACK_RIGHT",
#     ]

#     def test_inference(self):
#         """Inference test."""
#         cc_3dt = FasterRCNNCC3DT(
#             num_classes=len(nuscenes_track_map),
#             backbone="resnet50",
#             motion_model="KF3D",
#             pure_det=False,
#             class_range_map=torch.Tensor(nuscenes_class_range_map),
#             weights=self.model_weights,
#         )

#         data_root = osp.join(
#             get_test_data("nuscenes_test"), "track3d/images")
#         # TODO: It seems pylint will compain about sensors for transforms
#         preprocess_fn = compose(
#             [
#                 resize_image(  # pylint: disable=unexpected-keyword-arg
#                     shape=(900, 1600),
#                     keep_ratio=True,
#                     sensors=self.cameras,
#                 ),
#                 resize_intrinsics(  # pylint: disable=unexpected-keyword-arg
#                     sensors=self.cameras
#                 ),
#             ]
#         )

#         # TODO: add nuscenes test data
#         test_data = VideoDataPipe(
#             NuScenes(
#                 data_root,
#                 version="v1.0-mini",
#                 split="mini_val",
#                 metadata=["use_camera"],
#             ),
#             preprocess_fn=preprocess_fn,
#         )
#         batch_fn = compose(
#             [
#                 pad_image(  # pylint: disable=unexpected-keyword-arg
#                     sensors=self.cameras
#                 ),
#                 batched_normalize_image(  # pylint: disable=unexpected-keyword-arg, line-too-long
#                     sensors=self.cameras
#                 ),
#             ]
#         )
#         batch_size = 1
#         test_loader = build_inference_dataloaders(
#             test_data,
#             samples_per_gpu=batch_size,
#             workers_per_gpu=0,
#             batchprocess_fn=batch_fn,
#             collate_fn=multi_sensor_collate,
#         )[0]

#         data = next(iter(test_loader))
#         # assume: inputs are consecutive frames
#         images = []
#         inputs_hw = []
#         frame_ids = []
#         intrinscs = []
#         extrinsics = []
#         for cam in self.cameras:
#             images.append(data[cam][CommonKeys.images])
#             inputs_hw.extend(data[cam][CommonKeys.original_hw])
#             intrinscs.append(data[cam][CommonKeys.intrinsics])
#             extrinsics.append(data[cam][CommonKeys.extrinsics])
#             frame_ids.append(data[cam][CommonKeys.frame_ids])

#         images = torch.cat(images, dim=0)
#         intrinsics = torch.cat(intrinscs, dim=0)
#         extrinsics = torch.cat(extrinsics, dim=0)

#         cc_3dt.eval()
#         with torch.no_grad():
#             tracks = cc_3dt(  # pylint: disable=unused-variable
#                 images, inputs_hw, intrinsics, extrinsics, frame_ids
#             )

#         testcase_gt = torch.load(get_test_file("cc_3dt.pt"))
#         for pred, expected in zip(tracks, testcase_gt):
#             for pred_entry, expected_entry in zip(pred, expected):
#                 pass
#                 assert (
#                     torch.isclose(pred_entry, expected_entry, atol=1e-4)
#                     .all()
#                     .item()
#                 )

#     def test_train(self):
#         """Training test."""
#         pass

#     def test_torchscript(self):
#         """Test torchscipt export."""
#         sample_images = torch.rand((2, 3, 512, 512))
#         qdtrack = FasterRCNNQDTrack(num_classes=8)
#         qdtrack_scripted = torch.jit.script(qdtrack)
#         qdtrack_scripted(sample_images)
