#!/usr/bin/env bash
work_dir=$1
version=$2

#if version is mini, then use mini dataset
if [ $version == "mini" ]
then
    dataroot=vis4d/data/nuscenes_mini/ #for mini
elif [ $version == "test" ]
then
    dataroot=vis4d/data/nuscenes_test/ #for test
elif [ $version == "trainval" ]
then
    dataroot=vis4d/data/nuscenes/ #for trainval
fi


# 3D Detection Evaluation
# python vis4d-workspace/eval_nusc.py \
# --input $work_dir/detect_3d \
# --version v1.0-${version} \
# --dataroot $dataroot \
# --mode detection

# 3D Tracking Evaluation
python eval_nusc.py \
--input $work_dir/track_3d \
--version v1.0-${version} \
--dataroot=$dataroot \
--mode tracking
