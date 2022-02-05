#!/bin/bash
eval "$(conda shell.bash hook)"
conda create --name vis4d python=3.8
conda activate vis4d
conda install pytorch torchvision torchaudio cpuonly -c pytorch
python3 -m pip install --ignore-installed  -r requirements.txt
python3 -m pip install mmcv-full==1.4.0
python3 -m pip install git+https://github.com/facebookresearch/detectron2.git
python3 -m pip install git+https://github.com/SysCV/mmdetection.git
python3 -m pip install mmsegmentation
python3 -m pip install mmcls