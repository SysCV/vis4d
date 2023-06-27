#!/bin/bash
python3 -m pip install --ignore-installed -r requirements/install.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html
python3 -m pip install --ignore-installed -r requirements/dev.txt
