#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate VTL

python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=satellite_0_foxtechmapping_0 --datasets_folder ./datasets --infer_batch_size 256 --use_faiss_gpu --separate_branch

python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=satellite_0_foxtechmapping_0 --datasets_folder ./datasets --infer_batch_size 256 --prior_location_threshold=512 --use_faiss_gpu --separate_branch

python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=satellite_0_foxtechmapping_0 --datasets_folder ./datasets --infer_batch_size 256 --prior_location_threshold=512 --use_faiss_gpu --use_best_n=3 --separate_branch

python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=satellite_0_foxtechmapping_0 --datasets_folder ./datasets --infer_batch_size 256 --prior_location_threshold=512 --use_faiss_gpu --use_best_n=2 --separate_branch

python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=satellite_0_foxtechmapping_0 --datasets_folder ./datasets --infer_batch_size 256 --prior_location_threshold=512 --use_faiss_gpu --use_best_n=1 --separate_branch

python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=satellite_0_satellite_1 --datasets_folder ./datasets --infer_batch_size 256 --use_faiss_gpu --separate_branch

python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=satellite_0_satellite_1 --datasets_folder ./datasets --infer_batch_size 256 --prior_location_threshold=512 --use_faiss_gpu --separate_branch

python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=satellite_0_satellite_3 --datasets_folder ./datasets --infer_batch_size 256 --use_faiss_gpu --separate_branch

python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=satellite_0_satellite_3 --datasets_folder ./datasets --infer_batch_size 256 --prior_location_threshold=512 --use_faiss_gpu --separate_branch

python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=satellite_0_thermalmapping_012345 --datasets_folder ./datasets --infer_batch_size 256 --use_faiss_gpu --separate_branch

python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=satellite_0_thermalmapping_012345 --datasets_folder ./datasets --infer_batch_size 256 --prior_location_threshold=512 --use_faiss_gpu --separate_branch