#!/bin/bash
python3 -m pip install -r requirements/install.txt
python3 -m pip install -r requirements/torch-lib.txt --no-build-isolation --no-cache-dir
python3 -m pip install -r requirements/viewer.txt
python3 -m pip install -r requirements/dev.txt
