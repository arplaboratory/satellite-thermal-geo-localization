#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=eval_thermal

eval "$(conda shell.bash hook)"
conda activate VPR_SSL

python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=ADASI_0_ADASIthermal_0 --datasets_folder ./datasets --infer_batch_size 16 --use_faiss_gpu --backbone $2 --conv_output_dim 4096 --add_bn

python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=ADASI_0_ADASIthermal_0 --datasets_folder ./datasets --infer_batch_size 16 --prior_location_threshold=512 --use_faiss_gpu --backbone $2 --conv_output_dim 4096 --add_bn