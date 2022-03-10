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
