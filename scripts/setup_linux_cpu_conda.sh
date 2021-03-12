#!/bin/bash
eval "$(conda shell.bash hook)"
conda create --name openmt python=3.8
conda activate openmt
python -m pip install --ignore-installed  -r requirements.txt
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.7/index.html
