#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate VTL

echo "yea"
python3 train.py --dataset_name=satellite_0_satellite_1 --mining=partial --datasets_folder=datasets --infer_batch_size 256 --train_batch_size 12 --lr 0.0001