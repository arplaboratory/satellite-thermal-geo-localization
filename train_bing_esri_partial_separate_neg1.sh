#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate VTL

python3 train.py --dataset_name=satellite_0_satellite_1 --mining=partial --datasets_folder=datasets --infer_batch_size 128 --train_batch_size 48 --lr 0.0001 --separate_branch --negs_num_per_query 1