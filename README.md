# Machine learning (XGB) based bias correction for meps windspeed, windgust and temperature deterministic weather forecasts
This Machine Learning (ML) correction can be used to correct the forecast errors in deterministic weather model forecasts for 2-66 hour leadtimes. The input data is the model grids (grib2-files) and the output is corrected model fields (grib2). Machine learning correction is a gradient boosting regressor based on 2.5 years of training data and is calculated for points which are then further gridded back to NWP model background using Gridpp. ML correction is available for following parameters: 10m wind speed, 10m wind gust and 2m temperature.

## Usage
```
python3 xgb_predict_all.py --parameter windspeed --topography_data "meps_topography.grib" --landseacover_data "meps_lsm.grib" --fg_data "FFG-MS_10.grib2" --lcc_data "NL-0TO1_0.grib2" --mld_data "MIXHGT-M_0.grib2" --p_data "P-PA_0.grib2" --t2_data "T-K_2.grib2" --t850_data "T-K_850.grib2" --tke925_data "TKEN-JKG_925.grib2" --u10_data "U-MS_10.grib2" --u850_data "U-MS_850.grib2" --u65_data "U-MS_65.grib2" --v10_data "V-MS_10.grib2" --v850_data "V-MS_850.grib2" --v65_data "V-MS_65.grib2" --ugust_data "WGU-MS_10.grib2" --vgust_data "WGV-MS_10.grib2" --z500_data "Z-M2S2_500.grib2" --z1000_data "Z-M2S2_1000.grib2" --z0_data "Z-M2S2_0.grib2" --r2_data "RH-0TO1_2.grib2" --t0_data "T-K_0.grib2" --model "xgb_windspeed_20231214.json" --quantiles "quantiles_windspeed_20231214.npz" --station_list "all_stations_windspeed.csv" --producer_id "215" --output "out-FF-MS_10.grib2"
OR
./run_xgb_predict.sh YYYYMMDDHH parameter outfile.grib2 #parameter: windspeed, windgust, temperature
```

## Authors
kaisa.ylinen@fmi.fi
