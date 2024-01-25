#!/bin/bash
# Run predict script for the meps ml correction forecast 
#source ../../bin/activate
#E.g. bash run_xgb_predict.sh 2024012406 "windspeed" > log/log_run_xgb_predict 

PYTHON=python3
START_TIME=$1 #YYYYMMDDHH
PARAMETER=$2 #"windspeed", "windgust", "temperature"

echo "START_TIME:" $START_TIME
echo "PARAMETER:" $PARAMETER

bucket="s3://routines-data/meps-ml-correction/preop/"$START_TIME"00/"
bucket_model="s3://ml-models/meps-ml-correction/"

# Load data from S3
TOPO=$bucket_model"meps_topography.grib"
LC=$bucket_model"meps_lsm.grib"
FG=$bucket"FFG-MS_10.grib2"
LCC=$bucket"NL-0TO1_0.grib2"
MLD=$bucket"MIXHGT-M_0.grib2"
P0=$bucket"P-PA_0.grib2"
T2=$bucket"T-K_2.grib2"
T850=$bucket"T-K_850.grib2"
TKE925=$bucket"TKEN-JKG_925.grib2"
U10=$bucket"U-MS_10.grib2"
U850=$bucket"U-MS_850.grib2"
U65=$bucket"U-MS_65.grib2"
V10=$bucket"V-MS_10.grib2"
V850=$bucket"V-MS_850.grib2"
V65=$bucket"V-MS_65.grib2"
UGUST=$bucket"WGU-MS_10.grib2"
VGUST=$bucket"WGV-MS_10.grib2"
Z1000=$bucket"Z-M2S2_1000.grib2"
Z500=$bucket"Z-M2S2_500.grib2"
Z0=$bucket"Z-M2S2_0.grib2"
R2=$bucket"RH-0TO1_2.grib2"
T0=$bucket"T-K_0.grib2"
MODEL=$bucket_model"xgb_"$PARAMETER"_20231214.json"
QUANTILES=$bucket_model"quantiles_"$PARAMETER"_20231214.npz"
STATIONS=$bucket_model"all_stations_"$PARAMETER".csv"
OUTPUT="/data/statcal/projects/MEPS_WS_correction/forecasts/"$PARAMETER"_"$START_TIME".grib2"

#Use local data
#meps_folder = "/data/statcal/projects/MEPS_WS_correction/metcoopdata/"
#TOPO=$meps_folder"meps_topography.grib"
#LC=$meps_folder"meps_lsm.grib"
#FG=$meps_folder"FFG-MS_10.grib2"
#LCC=$meps_folder"NL-0TO1_0.grib2"
#MLD=$meps_folder"MIXHGT-M_0.grib2"
#P0=$meps_folder"P-PA_0.grib2"
#T2=$meps_folder"T-K_2.grib2"
#T850=$meps_folder"T-K_850.grib2"
#TKE925=$meps_folder"TKEN-JKG_925.grib2"
#U10=$meps_folder"U-MS_10.grib2"
#U850=$meps_folder"U-MS_850.grib2"
#U65=$meps_folder"U-MS_65.grib2"
#V10=$meps_folder"V-MS_10.grib2"
#V850=$meps_folder"V-MS_850.grib2"
#V65=$meps_folder"V-MS_65.grib2"
#UGUST=$meps_folder"WGU-MS_10.grib2"
#VGUST=$meps_folder"WGV-MS_10.grib2"
#Z1000=$meps_folder"Z-M2S2_1000.grib2"
#Z500=$meps_folder"Z-M2S2_500.grib2"
#Z0=$meps_folder"Z-M2S2_0.grib2"
#R2=$meps_folder"RH-0TO1_2.grib2"
#T0=$meps_folder"T-K_0.grib2"
#MODEL="/data/statcal/projects/MEPS_WS_correction/Models/xgb_"$PARAMETER"_20231214.json"
#QUANTILES="/data/statcal/projects/MEPS_WS_correction/Models/quantiles_"$PARAMETER"_20231214.npz"
#STATIONS="/data/statcal/projects/MEPS_WS_correction/trainingdata/all_stations_"$PARAMETER".csv"
#OUTPUT="/data/statcal/projects/MEPS_WS_correction/forecasts/"$PARAMETER"_"$START_TIME".grib2"


#Generating ml corrected forecast for parameter
$PYTHON xgb_predict_all.py --parameter $PARAMETER --topography_data $TOPO --landseacover_data $LC --fg_data $FG --lcc_data $LCC --mld_data $MLD --p_data $P0 --t2_data $T2 --t850_data $T850 --tke925_data $TKE925 --u10_data $U10 --u850_data $U850 --u65_data $U65 --v10_data $V10 --v850_data $V850 --v65_data $V65 --ugust_data $UGUST --vgust_data $VGUST --z500_data $Z500 --z1000_data $Z1000 --z0_data $Z0 --r2_data $R2 --t0_data $T0 --model $MODEL --quantiles $QUANTILES --station_list $STATIONS --output $OUTPUT --plot

# Uncomment Python run call for doing visualizations
# Check if "figures" is in project, if not, create
#mkdir -p "$PWD"/figures
# Generating visualizations for each forecasted timesteps
#$PYTHON plotting.py --input_file $OUTPUT
