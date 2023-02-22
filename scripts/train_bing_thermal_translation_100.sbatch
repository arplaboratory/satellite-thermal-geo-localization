#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64GB
#SBATCH --job-name=train_thermal
#SBATCH --output=train_thermal.out

eval "$(conda shell.bash hook)"
conda activate VTL

python3 train_pix2pix.py --dataset_name=satellite_0_thermalmapping_135 --datasets_folder=datasets --train_batch_size 16 --lr 0.0002 --patience 40 --epochs_num 40 --G_net unet_deep --G_loss_lambda 100.0 --G_gray --D_net patchGAN_deep --GAN_save_freq 5 --GAN_resize 1024 1024 --GAN_epochs_decay 20 --G_contrast