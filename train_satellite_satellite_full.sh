#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate VTL

python3 train.py --dataset_name=satellite_0_satellite_1 --mining=full --datasets_folder=datasets --infer_batch_size 128 --train_batch_size 10 --use_faiss_gpu --neg_samples_num 10000