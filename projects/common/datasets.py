"""Datasets used for projects in Vis4D."""
from vis4d.data.datasets import (
    BDD100K,
    COCO,
    KITTI,
    MOTChallenge,
    NuScenes,
    Scalabel,
    Waymo,
)

# CH, MOT17


mot17_map = {"pedestrian": 0}

crowdhuman_trainval = lambda: Scalabel(
    name="crowdhuman_trainval",
    annotations="data/CrowdHuman/trainval/trainval_scalabel.json",
    data_root="data/CrowdHuman/trainval",
    cache_as_binary=True,
)

mot17_train = lambda: MOTChallenge(
    name="mot17_train",
    data_root="data/MOT17/train",
    annotations="data/MOT17/train/train_scalabel.json",
    cache_as_binary=True,
)

mot17_val = lambda: MOTChallenge(
    name="mot17_val",
    data_root="data/MOT17/train",
    annotations="data/MOT17/train/val_scalabel.json",
    eval_metrics=["detect", "track"],
    cache_as_binary=True,
)

mot20_train = lambda: MOTChallenge(
    name="mot20_train",
    data_root="data/MOT20/train",
    annotations="data/MOT20/train/train_scalabel.json",
    cache_as_binary=True,
)

mot20_val = lambda: MOTChallenge(
    name="mot20_val",
    data_root="data/MOT20/train",
    annotations="data/MOT20/train/val_scalabel.json",
    eval_metrics=["detect", "track"],
    cache_as_binary=True,
)

# BDD100K
bdd100k_track_map = {
    "pedestrian": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "bicycle": 7,
}

bdd100k_track_train = lambda: BDD100K(
    name="bdd100k_track_train",
    annotations="data/bdd100k/labels/box_track_20/train/",
    data_root="data/bdd100k/images/track/train/",
    config_path="box_track",
    num_processes=0,
    cache_as_binary=True,
)

bdd100k_det_train = lambda: BDD100K(
    name="bdd100k_det_train",
    annotations="data/bdd100k/labels/det_20/det_train.json",
    data_root="data/bdd100k/images/100k/train/",
    config_path="det",
    num_processes=0,
    cache_as_binary=True,
)


bdd100k_track_val = lambda: BDD100K(
    name="bdd100k_track_val",
    annotations="data/bdd100k/labels/box_track_20/val/",
    data_root="data/bdd100k/images/track/val/",
    eval_metrics=["detect", "track"],
    num_processes=0,
    cache_as_binary=True,
)

# COCO

coco_train = lambda: COCO(
    name="coco_det_train",
    annotations="data/COCO/annotations/instances_train2017.json",
    data_root="data/COCO/train2017",
    cache_as_binary=True,
)

coco_val = lambda: COCO(
    name="coco_det_val",
    annotations="data/COCO/annotations/instances_val2017.json",
    data_root="data/COCO/val2017",
    eval_metrics=["detect"],
)

# KITTI

kitti_track_train = lambda: KITTI(
    name="kitti_track_train",
    annotations="data/KITTI/tracking_training.json",
    data_root="data/KITTI",
    input_dir="data/KITTI",
    split="training",
    data_type="tracking",
    output_dir="data/KITTI",
)

kitti_track_val = lambda: KITTI(
    name="kitti_track_val",
    annotations="data/KITTI/tracking_training.json",
    data_root="data/KITTI",
    input_dir="data/KITTI",
    split="training",
    data_type="tracking",
    output_dir="data/KITTI",
    multi_sensor_inference=False,
)

# NuScenes

nuscenes_train = lambda: NuScenes(
    name="nuscenes_train",
    data_root="data/nuscenes",
    version="v1.0-trainval",
    split="train",
    add_non_key=False,
    annotations="data/nuscenes/scalabel_train.json",
)

nuscenes_mini_train = lambda: NuScenes(
    name="nuscenes_mini_train",
    data_root="data/nuscenes",
    version="v1.0-mini",
    split="mini_train",
    add_non_key=False,
    annotations="data/nuscenes/scalabel_mini_train.json",
)

nuscenes_val = lambda: NuScenes(
    name="nuscenes_val",
    data_root="data/nuscenes",
    version="v1.0-trainval",
    split="val",
    add_non_key=True,
    annotations="data/nuscenes/scalabel_val_full.json",
    eval_metrics=["detect_3d", "track_3d"],
)

nuscenes_mini_val = lambda: NuScenes(
    name="nuscenes_mini_val",
    type="NuScenes",
    data_root="data/nuscenes",
    version="v1.0-mini",
    split="mini_val",
    add_non_key=True,
    annotations="data/nuscenes/scalabel_mini_val_full.json",
    eval_metrics=["detect_3d", "track_3d"],
)

# Waymo
waymo_train = lambda: Waymo(
    name="waymo_sample_train",
    data_root="data/Waymo_scalabel",
    input_dir="data/Waymo/train",
    output_dir="data/Waymo_scalabel",
)

waymo_val = lambda: Waymo(
    name="waymo_sample_val",
    data_root="data/Waymo_scalabel",
    input_dir="data/Waymo/train",
    output_dir="data/Waymo_scalabel",
    eval_metrics=["track"],
)
