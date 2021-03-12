#!/bin/bash
# python3 -m pip install --ignore-installed -r requirements.txt
# python3 -m pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 \
#   -f https://download.pytorch.org/whl/torch_stable.html
# python -m pip install detectron2 -f \
#   https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.7/index.html
python3 -m pip install -r requirements.txt \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html
python3 -m pip install 'git+git://github.com/facebookresearch/detectron2.git'