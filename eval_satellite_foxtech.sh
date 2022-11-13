#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate VTL

python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=satellite_0_foxtechmapping_0 --datasets_folder ./datasets --infer_batch_size 128 --use_faiss_gpu

python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=satellite_0_foxtechmapping_0 --datasets_folder ./datasets --infer_batch_size 128 --prior_location_threshold=512 --use_faiss_gpu