#!/bin/bash
#Script to run xgb training code 
#source ../../bin/activate
#E.g. bash run_xgb_train.sh > ../log/log_run_xgb_train

echo "run_xgb_train.sh alkaa"
date

variable="windspeed" #"windspeed", "windgust", "temperature" (lämpötilan osalta vaatii vielä joitain korjauksia koodiin)
first_year=2021
first_month=4
last_year=2023
last_month=9
model_name=$variable"_20231214"
training_data_dir='/data/statcal/projects/MEPS_WS_correction/trainingdata/'
model_dir="/data/statcal/projects/MEPS_WS_correction/Models/"

echo "Variable:" $variable
echo "Training period:" $first_month"/"$first_year"-"$last_month"/"$last_year
echo "XGB model run STARTS"

#activate python virtual environment
cd /home/users/ylinenk/Python3.9/xgb_model/
source ../bin/activate

python3 xgb_train_all.py --variable $variable --first_year $first_year --first_month $first_month --last_year $last_year --last_month $last_month --model_name $model_name --training_data_dir $training_data_dir --model_dir $model_dir > log/log_xgb_train_all_$model_name.txt

echo "XGB model run ENDS"
date
