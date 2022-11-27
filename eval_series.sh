#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate VTL

./eval_satellite_all.sh satellite_0_foxtechmapping_0-2022-11-26_09-18-32 resnet18conv4

./eval_satellite_all.sh satellite_0_satellite_1-2022-11-26_01-42-52 resnet18conv4

./eval_satellite_all.sh satellite_0_satellite_3-2022-11-26_05-04-19 resnet18conv4

./eval_satellite_all.sh satellite_0_thermalmapping_012345-2022-11-26_12-36-11 resnet18conv4
