#!/bin/bash
python3 -m pip install -r requirements.txt \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html
python3 -m pip install mmcv-full==1.4.5 \
    -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html
python3 -m pip install git+https://github.com/facebookresearch/detectron2.git
python3 -m pip install git+https://github.com/SysCV/mmdetection.git
python3 -m pip install mmsegmentation
python3 -m pip install mmcls