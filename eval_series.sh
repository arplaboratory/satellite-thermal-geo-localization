#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate VTL

./eval_satellite_foxtech.sh satellite_0_foxtechmapping_0-2022-11-16_05-59-29 resnet18conv4

./eval_satellite_satellite1.sh satellite_0_satellite_1-2022-11-15_23-53-02 resnet18conv4

./eval_satellite_satellite3.sh satellite_0_satellite_3-2022-11-16_02-52-18 resnet18conv4

./eval_satellite_thermal.sh satellite_0_thermalmapping_012345-2022-11-17_12-27-19 resnet18conv4

./eval_thermal_thermal.sh foxtechmapping_0_thermalmapping_012345-2022-11-17_20-37-38 resnet18conv4
