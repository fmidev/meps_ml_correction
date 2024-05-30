#!/bin/bash
#Script to run xgb training code 
#source ../../bin/activate
#E.g. bash run_xgb_train.sh "windgust" 20240304 > ../log/run_xgb_train.log

echo "run_xgb_train.sh alkaa"
date

variable=$1 #"windspeed", "windgust", "temperature", "dewpoint"
date_tag=$2
first_year=2021
first_month=4
last_year=2023
last_month=9
model_name=$variable"_"$date_tag
training_data_dir='/data/statcal/projects/MEPS_WS_correction/trainingdata/'
model_dir="/data/statcal/projects/MEPS_WS_correction/Models/"

echo "Variable:" $variable
echo "Training period:" $first_month"/"$first_year"-"$last_month"/"$last_year
echo "XGB model run STARTS"

python3 xgb_train_all.py --variable $variable --first_year $first_year --first_month $first_month --last_year $last_year --last_month $last_month --model_name $model_name --training_data_dir $training_data_dir --model_dir $model_dir > ../log/xgb_train_all_$model_name.log

echo "XGB model run ENDS"
date
