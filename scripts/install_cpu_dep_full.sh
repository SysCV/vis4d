#!/bin/bash
python3 -m pip install --ignore-installed -r requirements/base.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html
python3 -m pip install --ignore-installed -r requirements/optional.txt
python3 -m pip install --ignore-installed -r requirements/dev.txt
python3 -m pip install --ignore-installed -r requirements/visualization.txt
python3 -m pip install --ignore-installed -r requirements/datasets.txt
