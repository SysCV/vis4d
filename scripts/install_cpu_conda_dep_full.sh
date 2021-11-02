#!/bin/bash
eval "$(conda shell.bash hook)"
conda create --name vist python=3.8
conda activate vist
conda install pytorch torchvision torchaudio cpuonly -c pytorch
python3 -m pip install --ignore-installed  -r requirements.txt
python3 -m pip install mmcv-full
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
python3 -m pip install 'git+git://github.com/SysCV/mmdetection.git'
python3 -m pip install 'git+git://github.com/open-mmlab/mmsegmentation.git'