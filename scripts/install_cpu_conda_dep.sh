#!/bin/bash
eval "$(conda shell.bash hook)"
<<<<<<< HEAD
conda create --name openmt_tt python=3.8
conda activate openmt_tt
python3 -m pip install --ignore-installed  -r requirements.txt
=======
conda create --name vist python=3.8
conda activate vist
>>>>>>> c3cca4104d001b3f6a37227a0da755120a0a216b
conda install pytorch torchvision torchaudio cpuonly -c pytorch
python3 -m pip install --ignore-installed  -r requirements.txt
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
python3 -m pip install 'git+git://github.com/SysCV/mmdetection.git'
