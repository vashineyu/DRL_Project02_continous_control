#!/bin/sh

conda env create -f set_env.yaml
source activate tf
pip install ./python
python _environment_check.py

