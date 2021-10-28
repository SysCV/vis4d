#!/bin/bash
python3 -m pip install -r requirements.txt \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html
python3 -m pip install 'git+git://github.com/facebookresearch/detectron2.git'
python3 -m pip install 'git+git://github.com/SysCV/mmdetection.git'
python3 -m pip install 'git+git://github.com/open-mmlab/mmsegmentation.git'