#!/bin/bash
# Run predict script for the meps ml correction forecast 
#source ../../bin/activate
#E.g. bash run_xgb_predict.sh 2024012406 "windspeed" "path/to/file.grib2" > log/log_run_xgb_predict 

PYTHON=python3
ANALYSIS_TIME=$1 #YYYYMMDDHH
PARAMETER=$2 #"windspeed", "windgust", "temperature"
OUTPUT_FILE=$3

PRODUCER_ID=215

echo "ANALYSIS_TIME:" $ANALYSIS_TIME
echo "PARAMETER:" $PARAMETER
echo "OUTPUT_FILE:" $OUTPUT_FILE

# Load local (static) data
TOPO="meps_topography.grib"
LC="meps_lsm.grib"
if [ $PARAMETER="windspeed" ]
then
    MODEL="xgb_"$PARAMETER"_20231214.json"
    QUANTILES="quantiles_"$PARAMETER"_20231214.npz"
elif [  $PARAMETER="windgust" ]
     MODEL="xgb_"$PARAMETER"_20240304.json"
     QUANTILES="quantiles_"$PARAMETER"_20230304.npz"
elif [  $PARAMETER="windgust" ]
     MODEL="xgb_"$PARAMETER"_20240304.json"
     QUANTILES="quantiles_"$PARAMETER"_20230304.npz"
fi

STATIONS="all_stations_"$PARAMETER".csv"
   
# Load data from S3
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

#Define output file and path
#OUTPUT="/data/statcal/projects/MEPS_WS_correction/forecasts/"$PARAMETER"_"$ANALYSIS_TIME".grib2"

#If --plot argument is used create figures file
#mkdir -p figures

#Generating ml corrected forecast for parameter
$PYTHON xgb_predict_all.py --parameter $PARAMETER --topography_data $TOPO --landseacover_data $LC --fg_data $FG --lcc_data $LCC --mld_data $MLD --p_data $P0 --t2_data $T2 --t850_data $T850 --tke925_data $TKE925 --u10_data $U10 --u850_data $U850 --u65_data $U65 --v10_data $V10 --v850_data $V850 --v65_data $V65 --ugust_data $UGUST --vgust_data $VGUST --z500_data $Z500 --z1000_data $Z1000 --z0_data $Z0 --r2_data $R2 --t0_data $T0 --model $MODEL --quantiles $QUANTILES --station_list $STATIONS --producer_id $PRODUCER_ID --output $OUTPUT_FILE

