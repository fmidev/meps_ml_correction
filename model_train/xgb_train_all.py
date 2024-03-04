# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:21:16 2023

@author: ylinenk
"""
#Features from 4 grids are interpolated to station location

import os,sys,getopt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import math
import xgboost as xgb
import time
from sklearn.metrics import mean_squared_error
import quantile_mapping as qm  
from data_functions_1 import load_meps_training_data
from data_functions_1 import order_by_time_and_leadtime
from data_functions_1 import select_features
from data_functions_1 import add_time_lagged_features
from data_functions_1 import create_time_features
from data_functions_1 import create_station_features
from data_functions_1 import select_observation_variable
from data_functions_1 import rows_with_good_observations
from data_functions_1 import combine_all_features
from data_functions_1 import calculate_point_forecasts

###############################################
# LOAD AND MODIFY TRAINING DATA FOR XGB MODEL #
###############################################

options, remainder = getopt.getopt(sys.argv[1:],[],['variable=','first_year=','first_month=','last_year=','last_month=','model_name=','training_data_dir=','model_dir=','help'])
for opt, arg in options:
    if opt == '--help':
        print('xgb_train_all.py variable=windspeed, windgust or temperature, first_year=2021, first_month=4, last_year=2023, last_month=9, model_name=ws_xgb_model_20231211, training_data_dir=input data directory, model_dir=output data directory')
        exit()
    elif opt == '--variable': variable = arg
    elif opt == '--first_year': first_year = int(arg)
    elif opt == '--first_month': first_month = int(arg)
    elif opt == '--last_year': last_year = int(arg)
    elif opt == '--last_month': last_month = int(arg)
    elif opt == '--model_name': model_name = arg
    elif opt == '--training_data_dir': training_data_dir = arg
    elif opt == '--model_dir': model_dir = arg

if ((variable == "windspeed") | (variable == "temperature")):
    country_list = ['DEU','DNK', 'EST', 'FIN', 'LTU', 'LVA', 'NLD', 'POL', 'SWE']
elif (variable == "windgust"):
    country_list = ['DEU','EST','FIN','LVA','NLD','SWE'] #['DEU','DNK', 'EST', 'FIN', 'LTU', 'LVA', 'NLD', 'NOR', 'POL', 'SWE']
    
start = time.time()
print(time.ctime())

print("Variable:", variable)
print("Training period: " + str(first_month) + "/" + str(first_year) + "-" + str(last_month) + "/" + str(last_year) + "\n")
print("Data loading and modifying starts...")

#Load training data
features, labels, stations_all, metadata_all = load_meps_training_data(first_year, first_month, last_year, last_month, countries = country_list, main_directory = training_data_dir)

#Order by time and leadtime, and modify metadata, features and labels
features, labels, metadata_ordered = order_by_time_and_leadtime(metadata_all, features, labels)

#Select features that are used for our predicted variable
features = select_features(features, variable)

#Add lagged time features to features array
features, lt_ehto = add_time_lagged_features(metadata_ordered, features, n_lags=2)

#Create new time features
time_features = create_time_features(metadata_ordered, features.shape[1], lt_ehto)

#Create station specific features. WMO is used only to remove whole stations that do not have enough observations
station_features = create_station_features(stations_all, features.shape[0])

#Select observation variable from labels array
observations = select_observation_variable(labels, lt_ehto, variable)

#Select rows with good observations (no nans or stations that have too little observations)
true_rows = rows_with_good_observations(observations, station_features, variable)
all_observations = observations[true_rows]

print("Observations range:", np.min(all_observations),np.max(all_observations))

#Calculate point forecast and forecast error
forecasts_point = calculate_point_forecasts(features, true_rows, variable)
forecast_errors = forecasts_point - all_observations

#Combine all features
all_features = combine_all_features(features, time_features, station_features, true_rows)

print("Data modifing time:",str(math.floor((time.time() - start)/60)) + " minutes and " + str(round((time.time() - start) - math.floor((time.time() - start)/60)*60,1)) + " seconds")


###############################
## eXtreme Gradient boosting ##
###############################

#Tables for xgb model
x_trainval = all_features
y_trainval = forecast_errors

print("x_train array size for xgb model:", x_trainval.shape, "\n")
print("XGB training starts...")

if ((variable == "windspeed") | (variable == "windgust")):
    xgb_model = xgb.XGBRegressor(
            tree_method = "hist",
            n_estimators=504,#504,
            learning_rate=0.0117,
            max_depth=10,
            subsample=0.693,
            colsample_bytree=0.504,
            reg_alpha=0.714,
            objective='reg:squarederror')

elif (variable == "temperature"):
    xgb_model = xgb.XGBRegressor(
            tree_method = "hist",
            n_estimators=1204,#504,
            learning_rate=0.0117,
            max_depth=10,
            subsample=0.693,
            colsample_bytree=0.504,
            reg_alpha=0.714,
            objective='reg:squarederror')

start = time.time()
xgb_model.fit(x_trainval, y_trainval)
print("XGB model run time",str(math.floor((time.time() - start)/60)) + " minutes and " + str(round((time.time() - start) - math.floor((time.time() - start)/60)*60,1)) + " seconds")

xgb_trainval = xgb_model.predict(x_trainval)
predict_xgb_trainval = forecasts_point - xgb_trainval

if ((variable == "windspeed") | (variable == "windgust")):
    predict_xgb_trainval[predict_xgb_trainval < 0] = 0
#elif (variable == "temperature"):
#    predict_xgb_trainval[predict_xgb_trainval < 230] = 230

xgb_qm, q_obs, q_ctr = qm.q_mapping(all_observations,predict_xgb_trainval,predict_xgb_trainval,variable)

print("Model name:", model_name)
print("Saving models to folder", model_dir, "\n")

#Save model and qunatile arrays q_obs and q_ctr
xgb_model.save_model(model_dir + "xgb_" + model_name + ".json")
np.savez(model_dir + "quantiles_" + model_name + ".npz", q_obs=q_obs, q_ctr=q_ctr)
#plt.plot(q_obs,q_ctr)

###################
# END OF TRAINING #
###################

#XGB mallien korjaukset jakauma
predict_xgb_qm = predict_xgb_trainval.copy()
if ((variable == "windspeed") | (variable == "windgust")):
    predict_xgb_qm[predict_xgb_qm > 10] = qm.interp_extrap(x=predict_xgb_trainval[predict_xgb_qm > 10], xp=q_ctr, yp=q_obs)
#elif (variable == "temperature"):
    #predict_xgb_qm[predict_xgb_qm < 260] = qm.interp_extrap(x=predict_xgb_trainval[predict_xgb_qm < 260], xp=q_ctr, yp=q_obs)
xgb_qm_test = forecasts_point - predict_xgb_qm

print("Largest corrections for train set with xgb model:")
print(np.sort(xgb_trainval))
print("Largest corrections for train set with xgb_qm model:")
print(np.sort(xgb_qm_test))

print("\nMEPS mse:", round(mean_squared_error(all_observations, forecasts_point),3))
print("XGB mse:", round(mean_squared_error(all_observations, predict_xgb_trainval),3))
print("XGB_qm mse:", round(mean_squared_error(all_observations, predict_xgb_qm),3))

print(time.ctime())
print("END")
