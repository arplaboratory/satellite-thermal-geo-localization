#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate VPR_SSL

python3 train.py --dataset_name=satellite_0_satellite_1 --mining=full --datasets_folder=datasets