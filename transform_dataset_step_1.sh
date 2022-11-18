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

# # thermal_1 + thermal_1
# python h5_transformer.py --database_name thermalmapping --database_index 0 --queries_name thermalmapping --queries_index 0 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2

# # thermal_2 + thermal_2
# python h5_transformer.py --database_name thermalmapping --database_index 1 --queries_name thermalmapping --queries_index 1 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2

# # thermal_3 + thermal_3
# python h5_transformer.py --database_name thermalmapping --database_index 2 --queries_name thermalmapping --queries_index 2 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2

# # thermal_4 + thermal_4
# python h5_transformer.py --database_name thermalmapping --database_index 3 --queries_name thermalmapping --queries_index 3 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2

# # thermal_5 + thermal_5
# python h5_transformer.py --database_name thermalmapping --database_index 4 --queries_name thermalmapping --queries_index 4 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2

# # thermal_6 + thermal_6
# python h5_transformer.py --database_name thermalmapping --database_index 5 --queries_name thermalmapping --queries_index 5 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2

# # thermal_123456 + thermal_123456
# python h5_merger.py --database_name thermalmapping --database_indexes 012345 --queries_name thermalmapping --queries_indexes 012345 --compress --region_num 2

# rm -r ./datasets/thermalmapping_0_thermalmapping_0
# rm -r ./datasets/thermalmapping_1_thermalmapping_1
# rm -r ./datasets/thermalmapping_2_thermalmapping_2
# rm -r ./datasets/thermalmapping_3_thermalmapping_3
# rm -r ./datasets/thermalmapping_4_thermalmapping_4
# rm -r ./datasets/thermalmapping_5_thermalmapping_5

# foxtech + thermal_1
python h5_transformer.py --database_name foxtechmapping --database_index 0 --queries_name thermalmapping --queries_index 0 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2 &

# foxtech + thermal_2
python h5_transformer.py --database_name foxtechmapping --database_index 0 --queries_name thermalmapping --queries_index 1 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2 &

# foxtech + thermal_3
python h5_transformer.py --database_name foxtechmapping --database_index 0 --queries_name thermalmapping --queries_index 2 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2 &

# foxtech + thermal_4
python h5_transformer.py --database_name foxtechmapping --database_index 0 --queries_name thermalmapping --queries_index 3 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2 &

# foxtech + thermal_5
python h5_transformer.py --database_name foxtechmapping --database_index 0 --queries_name thermalmapping --queries_index 4 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2 &

# foxtech + thermal_6
python h5_transformer.py --database_name foxtechmapping --database_index 0 --queries_name thermalmapping --queries_index 5 --compress --train_sample_num 10000 --val_sample_num 10000 --region_num 2 &

# foxtech + thermal_123456
# python h5_merger.py --database_name foxtechmapping --database_indexes 0 --queries_name thermalmapping --queries_indexes 012345 --compress --region_num 2

# rm -r ./datasets/foxtechmapping_0_thermalmapping_0
# rm -r ./datasets/foxtechmapping_0_thermalmapping_1
# rm -r ./datasets/foxtechmapping_0_thermalmapping_2
# rm -r ./datasets/foxtechmapping_0_thermalmapping_3
# rm -r ./datasets/foxtechmapping_0_thermalmapping_4
# rm -r ./datasets/foxtechmapping_0_thermalmapping_5