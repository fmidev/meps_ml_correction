#!/bin/bash
# Run predict script for the meps ml correction forecast 
#source ../../bin/activate
#E.g. ./run_xgb_predict.sh 2024012406 "windspeed" "path/to/file.grib2" 215 > log/log_run_xgb_predict 

PYTHON=python3
ANALYSIS_TIME=$1 #YYYYMMDDHH
PARAMETER=$2 #"windspeed", "windgust", "temperature", "dewpoint"
OUTPUT_FILE=$3

#215 for preop, 214 for oper
PRODUCER_ID=$4 

echo "ANALYSIS_TIME:" $ANALYSIS_TIME
echo "PARAMETER:" $PARAMETER
echo "OUTPUT_FILE:" $OUTPUT_FILE

# Local (static) data
TOPO="meps_topography.grib"
LC="meps_lsm.grib"

# Model data,tag refers to model version, TAGs are defined in Containerfile
MODEL_WS="xgb_windspeed_"$WS_TAG".json"
QUANTILES_WS="quantiles_windspeed_"$WS_TAG".npz"
STATIONS_WS="all_stations_windspeed.csv"

MODEL_WG="xgb_windgust_"$WG_TAG".json"
QUANTILES_WG="quantiles_windgust_"$WG_TAG".npz"
STATIONS_WG="all_stations_windgust.csv"

MODEL_TA="xgb_temperature_"$TA_TAG".json"
QUANTILES_TA="quantiles_temperature_"$TA_TAG".npz"
STATIONS_TA="all_stations_temperature.csv"

MODEL_TD="xgb_dewpoint_"$TD_TAG".json"
QUANTILES_TD="quantiles_dewpoint_"$TD_TAG".npz"
STATIONS_TD="all_stations_dewpoint.csv"

MODEL_TMAX="xgb_t_max_"$TMAX_TAG".json"
QUANTILES_TMAX="quantiles_t_max_"$TMAX_TAG".npz"

MODEL_TMIN="xgb_t_min_"$TMIN_TAG".json"
QUANTILES_TMIN="quantiles_t_min_"$TMIN_TAG".npz"
   
# Data from S3
bucket="s3://routines-data/meps-ml-correction/preop/"$ANALYSIS_TIME"00/"
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
TD2=$bucket"TD-K_2.grib2"
TMAX=$bucket"TMAX-K_2.grib2"
TMIN=$bucket"TMIN-K_2.grib2"

#If --plot argument is used create figures file
#mkdir -p figures

#Generating ml corrected forecast for parameter
$PYTHON xgb_predict_all.py --parameter $PARAMETER --topography_data $TOPO --landseacover_data $LC --fg_data $FG --lcc_data $LCC --mld_data $MLD --p_data $P0 --t2_data $T2 --t850_data $T850 --tke925_data $TKE925 --u10_data $U10 --u850_data $U850 --u65_data $U65 --v10_data $V10 --v850_data $V850 --v65_data $V65 --ugust_data $UGUST --vgust_data $VGUST --z500_data $Z500 --z1000_data $Z1000 --z0_data $Z0 --r2_data $R2 --t0_data $T0 --td2_data $TD2 --tmax_data $TMAX --tmin_data $TMIN --model_ws $MODEL_WS --quantiles_ws $QUANTILES_WS --model_wg $MODEL_WG --quantiles_wg $QUANTILES_WG --model_ta $MODEL_TA --quantiles_ta $QUANTILES_TA --model_td $MODEL_TD --quantiles_td $QUANTILES_TD --model_tmax $MODEL_TMAX --quantiles_tmax $QUANTILES_TMAX --model_tmin $MODEL_TMIN --quantiles_tmin $QUANTILES_TMIN --stations_list_ws $STATIONS_WS --stations_list_wg $STATIONS_WG --stations_list_ta $STATIONS_TA --stations_list_td $STATIONS_TD --analysis_time $ANALYSIS_TIME --producer_id $PRODUCER_ID --output $OUTPUT_FILE 


