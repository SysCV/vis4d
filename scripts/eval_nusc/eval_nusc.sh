#!/usr/bin/env bash
work_dir=$1
version=$2

# 3D Detection Evaluation
python eval_nusc.py \
--input $work_dir/detect_3d \
--version v1.0-${version} \
--dataroot data/nuscenes \
--mode detection

# 3D Tracking Evaluation
python eval_nusc.py \
--input $work_dir/track_3d \
--version v1.0-${version} \
--dataroot=data/nuscenes \
--mode tracking
