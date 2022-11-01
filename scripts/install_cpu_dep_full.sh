#!/bin/bash
python3 -m pip install -r requirements/base.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html
# python3 -m pip install -r requirements/optional.txt
python3 -m pip install -r requirements/dev.txt
python3 -m pip install -r requirements/visualization.txt
