#!/bin/bash
#Script to run xgb training code 
#source ../../bin/activate
#E.g. bash run_xgb_train.sh "windgust" 20240304 > ../log/run_xgb_train.log

echo "run_xgb_train.sh alkaa"
date

variable=$1 #"windspeed", "windgust", "temperature", "dewpoint", "t_max", "t_min"
date_tag=$2
first_year=2023
first_month=3
last_year=2026
last_month=2
training_data_dir='/home/users/ylinenk/projects/MEPS_WS_correction/trainingdata/'
model_dir="/home/users/ylinenk/projects/MEPS_WS_correction/Models/"

forecast_errors=true

echo "Variable:" $variable
echo "Training period:" $first_month"/"$first_year"-"$last_month"/"$last_year
echo "XGB model run STARTS"

if [ "$forecast_errors" = true ]; then
    model_name=$variable"_"$date_tag"_forecast_errors_4"
    python3 xgb_train_all_forecast_errors.py --variable $variable --first_year $first_year --first_month $first_month --last_year $last_year --last_month $last_month --model_name $model_name --training_data_dir $training_data_dir --model_dir $model_dir > ../log/xgb_train_all_$variable"_"$date_tag.log
else
    model_name=$variable"_"$date_tag"_default_4"
    python3 xgb_train_all.py --variable $variable --first_year $first_year --first_month $first_month --last_year $last_year --last_month $last_month --model_name $model_name --training_data_dir $training_data_dir --model_dir $model_dir > ../log/xgb_train_all_$variable"_"$date_tag.log
fi

echo "XGB model run ENDS"
date
