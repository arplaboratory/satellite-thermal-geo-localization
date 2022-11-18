#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate VPR_SSL

# bing + thermal_123456
python h5_merger.py --database_name satellite --database_indexes 0 --queries_name thermalmapping --queries_indexes 012345 --compress --region_num 2

rm -r ./datasets/satellite_0_thermalmapping_0
rm -r ./datasets/satellite_0_thermalmapping_1
rm -r ./datasets/satellite_0_thermalmapping_2
rm -r ./datasets/satellite_0_thermalmapping_3
rm -r ./datasets/satellite_0_thermalmapping_4
rm -r ./datasets/satellite_0_thermalmapping_5

# # thermal_123456 + thermal_123456
# python h5_merger.py --database_name thermalmapping --database_indexes 012345 --queries_name thermalmapping --queries_indexes 012345 --compress --region_num 2

# rm -r ./datasets/thermalmapping_0_thermalmapping_0
# rm -r ./datasets/thermalmapping_1_thermalmapping_1
# rm -r ./datasets/thermalmapping_2_thermalmapping_2
# rm -r ./datasets/thermalmapping_3_thermalmapping_3
# rm -r ./datasets/thermalmapping_4_thermalmapping_4
# rm -r ./datasets/thermalmapping_5_thermalmapping_5

# foxtech + thermal_123456
python h5_merger.py --database_name foxtechmapping --database_indexes 0 --queries_name thermalmapping --queries_indexes 012345 --compress --region_num 2

rm -r ./datasets/foxtechmapping_0_thermalmapping_0
rm -r ./datasets/foxtechmapping_0_thermalmapping_1
rm -r ./datasets/foxtechmapping_0_thermalmapping_2
rm -r ./datasets/foxtechmapping_0_thermalmapping_3
rm -r ./datasets/foxtechmapping_0_thermalmapping_4
rm -r ./datasets/foxtechmapping_0_thermalmapping_5