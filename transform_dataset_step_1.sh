#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate VPR_SSL

# bing + foxtech
python h5_transformer.py --database_name satellite --database_index 0 --queries_name foxtechmapping --queries_index 0 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2 &

# bing + esri
python h5_transformer.py --database_name satellite --database_index 0 --queries_name satellite --queries_index 1 --compress --train_sample_num 40000 --val_sample_num 20000 --region_num 3 &

# bing + google
python h5_transformer.py --database_name satellite --database_index 0 --queries_name satellite --queries_index 3 --compress --train_sample_num 40000 --val_sample_num 20000 --region_num 3 &

# bing + thermal_1
python h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 0 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2 &

# bing + thermal_2
python h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 1 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2 &

# bing + thermal_3
python h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 2 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2 &

# bing + thermal_4
python h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 3 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2 &

# bing + thermal_5
python h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 4 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2 &

# bing + thermal_6
python h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 5 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2 &

# bing + thermal_123456
# python h5_merger.py --database_name satellite --database_indexes 0 --queries_name thermalmapping --queries_indexes 012345 --compress --region_num 2

# rm -r ./datasets/satellite_0_thermalmapping_0
# rm -r ./datasets/satellite_0_thermalmapping_1
# rm -r ./datasets/satellite_0_thermalmapping_2
# rm -r ./datasets/satellite_0_thermalmapping_3
# rm -r ./datasets/satellite_0_thermalmapping_4
# rm -r ./datasets/satellite_0_thermalmapping_5