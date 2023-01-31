"""CC-3DT Tests."""
import unittest
import json
import torch
from vis4d.engine.ckpt import load_model_checkpoint

from vis4d.data.const import CommonKeys
from vis4d.data.datasets.nuscenes import NuScenes
from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    multi_sensor_collate,
)
from vis4d.data.transforms.normalize import normalize_image
from vis4d.data.transforms.pad import pad_image
from vis4d.data.transforms.resize import resize_image

from vis4d.model.track3d.cc_3dt import FasterRCNNCC3DT
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from nuscenes.utils.data_classes import Quaternion
from vis4d.common.imports import is_torch_tf32_available
import pdb

nuscenes_track_map = {
    0: "bicycle",
    1: "motorcycle",
    2: "pedestrian",
    3: "bus",
    4: "car",
    5: "trailer",
    6: "truck",
    7: "construction_vehicle",
    8: "traffic_cone",
    9: "barrier",
}

DefaultAttribute = {
    "car": "vehicle.parked",
    "pedestrian": "pedestrian.moving",
    "trailer": "vehicle.parked",
    "truck": "vehicle.parked",
    "bus": "vehicle.moving",
    "motorcycle": "cycle.without_rider",
    "construction_vehicle": "vehicle.parked",
    "bicycle": "cycle.without_rider",
    "barrier": "",
    "traffic_cone": "",
}

tracking_cats = [
    "bicycle",
    "motorcycle",
    "pedestrian",
    "bus",
    "car",
    "trailer",
    "truck",
]


def get_attributes(name: str, velocity, velocity_thres: float = 0.2) -> str:
    """Get nuScenes attributes."""
    if np.sqrt(velocity[0] ** 2 + velocity[1] ** 2) > velocity_thres:
        if name in [
            "car",
            "construction_vehicle",
            "bus",
            "truck",
            "trailer",
        ]:
            attr = "vehicle.moving"
        elif name in ["bicycle", "motorcycle"]:
            attr = "cycle.with_rider"
        else:
            attr = DefaultAttribute[name]
    else:
        if name in ["pedestrian"]:
            attr = "pedestrian.standing"
        elif name in ["bus"]:
            attr = "vehicle.stopped"
        else:
            attr = DefaultAttribute[name]
    return attr


# class CC3DTTest(unittest.TestCase):
#     """CC-3DT class tests."""

#     model_weights = (
#         # "https://dl.cv.ethz.ch/vis4d/qdtrack_bdd100k_frcnn_res50_heavy_augs.pt"
#         "./vis4d-workspace/checkpoints/last.ckpt"
#     )

#     def test_inference(self):
def test_inference(attr_by_velocity: bool = False):
    """Inference test.

    Run::
        >>> pytest tests/model/track3d/cc_3dt_test.py::CC3DTTest::test_inference
    """
    if is_torch_tf32_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    model_weights = (
        # "https://dl.cv.ethz.ch/vis4d/qdtrack_bdd100k_frcnn_res50_heavy_augs.pt"
        # "./vis4d-workspace/checkpoints/last.ckpt"
        "./vis4d-workspace/checkpoints/qd_3dt_R_50_FPN_nuscenes_12_accumulate_gradient_2.ckpt"
    )

    cc_3dt = FasterRCNNCC3DT(num_classes=10)
    cc_3dt.cuda()
    load_model_checkpoint(cc_3dt, model_weights)

    test_data = DataPipe(
        NuScenes(
            "data/nuscenes/",
            version="v1.0-mini",
            split="mini_val",
            metadata=["use_camera"],
        ),
        # preprocess_fn=normalize_image(sensors=NuScenes._CAMERAS),
    )

    batch_fn = pad_image(sensors=NuScenes._CAMERAS)
    batch_size = 1
    test_loader = build_inference_dataloaders(
        test_data,
        samples_per_gpu=batch_size,
        workers_per_gpu=0,
        batchprocess_fn=batch_fn,
        collate_fn=multi_sensor_collate,
    )[0]

    # data = next(iter(test_loader))
    cc_3dt.eval()
    with torch.no_grad():
        results: DictStrAny = {}
        for i, data in enumerate(tqdm(test_loader)):
            # assume: inputs are consecutive frames
            annos = []
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

            frame_ids.extend(data[cam][CommonKeys.frame_ids])

            images = torch.cat(images, dim=0).cuda()
            intrinsics = torch.cat(intrinscs, dim=0).cuda()
            extrinsics = torch.cat(extrinsics, dim=0).cuda()

            tracks = cc_3dt(
                images, inputs_hw, intrinsics, extrinsics, frame_ids
            )

            pdb.set_trace()

            token = data[cam]["token"][0]

            if len(tracks.boxes_3d) != 0:
                for track_id, box_3d, score_3d, class_id in zip(
                    tracks.track_ids,
                    tracks.boxes_3d,
                    tracks.scores_3d,
                    tracks.class_ids,
                ):
                    category = nuscenes_track_map[int(class_id.cpu().numpy())]
                    if not category in tracking_cats:
                        continue

                    translation = box_3d[0:3].cpu().numpy()

                    dim = box_3d[3:6].cpu().numpy().tolist()
                    dimension = [dim[1], dim[2], dim[0]]
                    dimension = [d if d >= 0 else 0.1 for d in dimension]

                    # Using extrinsic rotation here to align with Pytorch3D
                    x, y, z, w = R.from_euler(
                        "XYZ", box_3d[6:9].cpu().numpy()
                    ).as_quat()
                    rotation = Quaternion([w, x, y, z])

                    score = float(score_3d.cpu().numpy())

                    velocity = box_3d[9:12].cpu().numpy().tolist()

                    if attr_by_velocity:
                        attribute_name = get_attributes(category, velocity)
                    else:
                        attribute_name = DefaultAttribute[category]

                    # nusc_anno = {
                    #     "sample_token": token,
                    #     "translation": translation.tolist(),
                    #     "size": dimension,
                    #     "rotation": rotation.elements.tolist(),
                    #     "velocity": [velocity[0], velocity[1]],
                    #     "detection_name": category,
                    #     "detection_score": score,
                    #     "attribute_name": attribute_name,
                    # }
                    nusc_anno = {
                        "sample_token": token,
                        "translation": translation.tolist(),
                        "size": dimension,
                        "rotation": rotation.elements.tolist(),
                        "velocity": [velocity[0], velocity[1]],
                        "tracking_id": int(track_id.cpu().numpy()),
                        "tracking_name": category,
                        "tracking_score": score,
                    }
                    annos.append(nusc_anno)
            results[token] = annos

    metadata = {
        "use_camera": True,
        "use_lidar": False,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    nusc_annos = {
        "results": results,
        "meta": metadata,
    }

    with open(
        # "vis4d-workspace/nusc_test/detect_3d_predictions.json",
        # mode="w",
        # encoding="utf-8"
        "vis4d-workspace/nusc_test/track_3d_predictions.json",
        mode="w",
        encoding="utf-8",
    ) as f:
        json.dump(nusc_annos, f)


test_inference()
