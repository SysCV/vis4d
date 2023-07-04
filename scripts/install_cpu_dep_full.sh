#!/bin/bash
python3 -m pip install -r requirements/install.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html
python3 -m pip install -r requirements/torch-lib.txt
python3 -m pip install -r requirements/viewer.txt
python3 -m pip install -r requirements/dev.txt
