#!/usr/bin/env bash

#dataset path as arg
data_dir=$1

docker run -it \
    --gpus all \
    --volume $data_dir/v1.0-trainval:/root/cc3dt/vis4d/data/nuscenes \
    --volume $data_dir/v1.0-mini:/root/cc3dt/vis4d/data/nuscenes_mini \
    --volume $data_dir/v1.0-test:/root/cc3dt/vis4d/data/nuscenes_test \
    --volume $data_dir/checkpoints:/root/cr3dt/checkpoints \
    --name="main_cc3dtpp_track" \
    --detach \
    --shm-size=1gb \
    cc3dt \
    /bin/bash
