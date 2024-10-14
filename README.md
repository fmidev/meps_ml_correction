# Machine learning (XGB) based bias correction for meps windspeed, windgust, temperature and dewpoint deterministic weather forecasts
This Machine Learning (ML) correction code can be used to correct the forecast errors in deterministic weather model forecasts for 2-66 hour leadtimes. The input data is the model grids (grib2-files) and the output is corrected model fields (grib2). Machine learning correction is an eXtreme Gradient Boosting (XGBoost) regressor based on 3 years of training data and is calculated for points which are then further gridded back to NWP model background using Gridpp. ML correction is available for following parameters: 10m wind speed, 10m wind gust, 2m temperature and 2m dewpoint temperature. 

For 2m temperature, the prediction code makes also min/max temperature combination. Which means that:
- when 2m temperature is lowest at night time (19-6 UTC), the 2m temperature forecast is replaced with 12 hours minimum temperature forecast
- when 2m temperature is highest at day time (7-18 UTC), the 2m temperature forecast is replaced with 12 hours maximum temperature forecasts

For 2m dewpoint temparture forecast and 10m wind gust forecast, the predction code checks that:
 - 2m dewpoint temparture forecast can not be higher than 2m temperature forecast at any grid point
 - wind gust forecast can not be lower than wind speed forecast at any grid point

## Usage
Running with run_xgb_predict.sh shell script:
```
./run_xgb_predict.sh YYYYMMDDHH parameter outfile.grib2 producer_id 
```
E.g. 
```
./run_xgb_predict.sh 2024101403 "windspeed" "windspeed_2024101403.grib2" 215
```

## Authors
kaisa.ylinen@fmi.fi
