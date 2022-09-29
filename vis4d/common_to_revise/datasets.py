"""Datasets used for projects in Vis4D."""
from vis4d.data_to_revise.datasets import (
    BDD100K,
    COCO,
    KITTI,
    MOTChallenge,
    NuScenes,
    Scalabel,
    Waymo,
)

# CH, MOT17, MOT20

mot_map = {"pedestrian": 0}

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
bdd100k_panseg_map = {
    "dynamic": 0,
    "ego vehicle": 1,
    "ground": 2,
    "static": 3,
    "parking": 4,
    "rail track": 5,
    "road": 6,
    "sidewalk": 7,
    "bridge": 8,
    "building": 9,
    "fence": 10,
    "garage": 11,
    "guard rail": 12,
    "tunnel": 13,
    "wall": 14,
    "banner": 15,
    "billboard": 16,
    "lane divider": 17,
    "parking sign": 18,
    "pole": 19,
    "polegroup": 20,
    "street light": 21,
    "traffic cone": 22,
    "traffic device": 23,
    "traffic light": 24,
    "traffic sign": 25,
    "traffic sign frame": 26,
    "terrain": 27,
    "vegetation": 28,
    "sky": 29,
    "person": 30,
    "rider": 31,
    "bicycle": 32,
    "bus": 33,
    "car": 34,
    "caravan": 35,
    "motorcycle": 36,
    "trailer": 37,
    "train": 38,
    "truck": 39,
}

bdd100k_base_path = "vis4d/engine_to_revise/testcases"
bdd100k_detect_sample = lambda: Scalabel(
    name="bdd100k_detect_sample",
    data_root=f"{bdd100k_base_path}/detect/bdd100k-samples/images",
    annotations=f"{bdd100k_base_path}/detect/bdd100k-samples/labels/",
    config_path=f"{bdd100k_base_path}/detect/bdd100k-samples/config.toml",
)

bdd100k_insseg_sample = lambda: Scalabel(
    name="bdd100k_insseg_sample",
    data_root=f"{bdd100k_base_path}/detect/bdd100k-samples/images",
    annotations=f"{bdd100k_base_path}/detect/bdd100k-samples/labels/",
    config_path=f"{bdd100k_base_path}/detect/bdd100k-samples/insseg_config.toml",
)

bdd100k_track_sample = lambda: Scalabel(
    name="bdd100k_track_sample",
    data_root=f"{bdd100k_base_path}/track/bdd100k-samples/images",
    annotations=f"{bdd100k_base_path}/track/bdd100k-samples/labels/",
    config_path=f"{bdd100k_base_path}/track/bdd100k-samples/config.toml",
)

bdd100k_segtrack_sample = lambda: Scalabel(
    name="bdd100k_segtrack_sample",
    data_root=f"{bdd100k_base_path}/track/bdd100k-samples/images",
    annotations=f"{bdd100k_base_path}/track/bdd100k-samples/labels/",
    config_path=f"{bdd100k_base_path}/track/bdd100k-samples/config.toml",
)

bdd100k_panseg_sample = lambda: BDD100K(
    name="bdd100k_panseg_sample",
    data_root=f"{bdd100k_base_path}/detect/bdd100k-samples/images",
    annotations=f"{bdd100k_base_path}/panoptic/bdd100k-samples/labels/",
    config_path="pan_seg",
)


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

bdd100k_sem_seg_train = lambda: BDD100K(
    name="bdd100k_sem_seg_train",
    annotations="data/bdd100k/labels/sem_seg_train_rle.json",
    data_root="data/bdd100k/images/10k/train/",
    config_path="sem_seg",
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

bdd100k_sem_seg_val = lambda: BDD100K(
    name="bdd100k_sem_seg_val",
    annotations="data/bdd100k/labels/sem_seg_val_rle.json",
    data_root="data/bdd100k/images/10k/val/",
    config_path="sem_seg",
    eval_metrics=["sem_seg"],
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

coco_insseg_val = lambda: COCO(
    name="coco_insseg_val",
    annotations="data/COCO/annotations/instances_val2017.json",
    data_root="data/COCO/val2017",
    eval_metrics=["detect", "ins_seg"],
)

# KITTI

kitti_track_map = {
    "car": 0,
    "pedestrian": 1,
    "cyclist": 2,
    "truck": 3,
    "tram": 4,
    "misc": 5,
}

kitti_track_train = lambda: KITTI(
    name="kitti_track_train",
    annotations="data/KITTI/tracking_training.json",
    data_root="data/KITTI",
    input_dir="data/KITTI",
    split="training",
    data_type="tracking",
    output_dir="data/KITTI",
)

kitti_det_train = lambda: KITTI(
    name="kitti_det_train",
    annotations="data/KITTI/detection_training.json",
    data_root="data/KITTI",
    data_type="detection",
    split="training",
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

nuscenes_track_map = {
    "bicycle": 0,
    "motorcycle": 1,
    "pedestrian": 2,
    "bus": 3,
    "car": 4,
    "trailer": 5,
    "truck": 6,
    "construction_vehicle": 7,
    "traffic_cone": 8,
    "barrier": 9,
}

nuscenes_train = lambda: NuScenes(
    name="nuscenes_train",
    data_root="data/nuscenes",
    version="v1.0-trainval",
    split="train",
    add_non_key=False,
    annotations="data/nuscenes/scalabel_train.json",
    cache_as_binary=True,
)

nuscenes_mini_train = lambda: NuScenes(
    name="nuscenes_mini_train",
    data_root="data/nuscenes",
    version="v1.0-mini",
    split="mini_train",
    add_non_key=False,
    annotations="data/nuscenes/scalabel_mini_train.json",
    cache_as_binary=True,
)

nuscenes_val = lambda: NuScenes(
    name="nuscenes_val",
    data_root="data/nuscenes",
    version="v1.0-trainval",
    split="val",
    add_non_key=False,
    annotations="data/nuscenes/scalabel_val.json",
    custom_save=True,
    eval_metrics=["detect_3d"],
    cache_as_binary=True,
)

nuscenes_mini_val = lambda: NuScenes(
    name="nuscenes_mini_val",
    data_root="data/nuscenes",
    version="v1.0-mini",
    split="mini_val",
    add_non_key=False,
    annotations="data/nuscenes/scalabel_mini_val.json",
    custom_save=True,
    eval_metrics=["detect_3d"],
    cache_as_binary=True,
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
