#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate VTL

python3 eval_pix2pix_generate_h5.py --resume='logs/default/'$1'/last_model.pth' --dataset_name=satellite_0_thermalmapping_135 --datasets_folder ./datasets --G_net unet_deep --G_gray --GAN_upsample bilinear --GAN_resize 1024 1024 --G_contrast