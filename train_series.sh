#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate VTL

./train_bing_esri_full.sh

./train_bing_google_full.sh

./train_bing_foxtech_full.sh

./train_bing_thermal_full.sh
 
./train_bing_esri_partial.sh

./train_bing_google_partial.sh 

./train_bing_foxtech_partial.sh

./train_bing_thermal_partial.sh 

./train_bing_esri_partial_separate.sh

./train_bing_google_partial_separate.sh 

./train_bing_foxtech_partial_separate.sh

./train_bing_thermal_partial_separate.sh 

./train_bing_esri_partial_separate_neg1.sh

./train_bing_google_partial_separate_neg1.sh 

./train_bing_foxtech_partial_separate_neg1.sh

./train_bing_thermal_partial_separate_neg1.sh 