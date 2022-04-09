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

bdd100k_det_map = {
    "pedestrian": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "traffic light": 8,
    "traffic sign": 9,
}
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
bdd100k_seg_map = {
    "road": 0,
    "sidewalk": 1,
    "building": 2,
    "wall": 3,
    "fence": 4,
    "pole": 5,
    "traffic light": 6,
    "traffic sign": 7,
    "vegetation": 8,
    "terrain": 9,
    "sky": 10,
    "pedestrian": 11,
    "rider": 12,
    "car": 13,
    "truck": 14,
    "bus": 15,
    "train": 16,
    "motorcycle": 17,
    "bicycle": 18,
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

bdd100k_pan_seg_train = lambda: BDD100K(
    name="bdd100k_pan_seg_train",
    annotations="data/bdd100k/labels/sem_ins_seg_train.json",
    data_root="data/bdd100k/images/10k/train/",
    config_path="pan_seg",
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

bdd100k_det_val = lambda: BDD100K(
    name="bdd100k_det_val",
    annotations="data/bdd100k/labels/det_20/det_val.json",
    data_root="data/bdd100k/images/100k/val/",
    config_path="det",
    eval_metrics=["detect"],
    num_processes=0,
    cache_as_binary=True,
)

bdd100k_pan_seg_val = lambda: BDD100K(
    name="bdd100k_pan_seg_val",
    annotations="data/bdd100k/labels/sem_ins_seg_val.json",
    data_root="data/bdd100k/images/10k/val/",
    config_path="pan_seg",
    eval_metrics=["pan_seg"],
    num_processes=0,
    cache_as_binary=True,
)

# COCO

coco_det_map = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "fire hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "frisbee": 29,
    "skis": 30,
    "snowboard": 31,
    "sports ball": 32,
    "kite": 33,
    "baseball bat": 34,
    "baseball glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
    "dining table": 60,
    "toilet": 61,
    "tv": 62,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "microwave": 68,
    "oven": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72,
    "book": 73,
    "clock": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77,
    "hair drier": 78,
    "toothbrush": 79,
}

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
