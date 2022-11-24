#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate VTL

# bing + thermal_123456
python h5_merger.py --database_name satellite --database_indexes 0 --queries_name thermalmapping --queries_indexes 012345 --compress --region_num 2 &

rm -r ./datasets/satellite_0_thermalmapping_0
rm -r ./datasets/satellite_0_thermalmapping_1
rm -r ./datasets/satellite_0_thermalmapping_2
rm -r ./datasets/satellite_0_thermalmapping_3
rm -r ./datasets/satellite_0_thermalmapping_4
rm -r ./datasets/satellite_0_thermalmapping_5