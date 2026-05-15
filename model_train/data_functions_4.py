# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:22:33 2023

@author: ylinenk
"""

from numpy import load
import numpy as np
import pandas as pd

#Load training data for all 4 closest grid points
def load_meps_training_data_4(first_year, first_month, last_year, last_month, countries, main_directory):
    """      
    This function loads the training data between first and last month/year for countries mentioned in the list.
    features values are interpolated to station points using 4 grid point values
    Returns two arrays: features and labels, and two lists: stations_all and metadata_all 
    Arguments:
        - first_year: int, e.g. 2021
        - first_month: int, e.g. 4
        - last_year: int, e.g. 2023
        - last_month: int, e.g. 10
        - countries: list of countries, e.g. ['EST', 'FIN', 'SWE']
        - main_directory: string, directory where training data exists
    """
    #Some basic checks
    if (first_year > last_year): raise ValueError('first_year should be smaller or equal than last_year.')
    
    #Load training data for all countries and months
    stations_list = []
    metadata_list = []

    luku = 0
    for maa in countries: #['DEU', 'DNK', 'EST', 'FIN', 'LTU', 'LVA', 'NLD', 'NOR', 'POL', 'SWE']
        maa_dir = (main_directory + maa)
        station_list_country = pd.read_csv(maa_dir + '/stations2.csv') #Z0 ja lsm added to stations2 list
        stations_list.append(station_list_country)

        features_country = np.empty((0, len(station_list_country), 22, 4), dtype=np.float32)
        labels_country = np.empty((0, len(station_list_country), 7), dtype=np.float32)

        for yyyy in list(range(first_year,last_year+1)):
            for mm in [1,2,3,4,5,6,7,8,9,10,11,12]:
                if ((yyyy == first_year) & (mm < first_month)): continue
                if ((yyyy == last_year) & (mm > last_month)): break
                #print(str(yyyy) + "/" + str(mm))

                data = load(maa_dir + "/" + str(yyyy) + '/' + str(mm) + '/trainingdata.npz', allow_pickle=True)

                if luku==0: #metadata is same for all countries
                    metadata = pd.read_csv(maa_dir + "/" + str(yyyy) + '/' + str(mm) + '/metadata.csv')
                    metadata_list.append(metadata)

                features_part = data['features']
                labels_part = data['lables'] #Obs! lables is not a typo, the file is named like that, but it contains labels data
                
                #Order features based on weights (highest weight first)
                features_part1 = order_by_weights(station_list_country, features_part)

                features_country = np.concatenate((features_country, features_part1), axis=0, dtype=np.float32)
                labels_country = np.concatenate((labels_country, labels_part), axis=0, dtype=np.float32)
        
        if luku==0:
            features = features_country
            labels = labels_country
        else:  
            features = np.concatenate((features, features_country), axis=1, dtype=np.float32)
            labels = np.concatenate((labels, labels_country), axis=1, dtype=np.float32)
        luku += 1

    stations_all = pd.concat(stations_list, ignore_index=True)
    metadata_all = pd.concat(metadata_list, ignore_index=True)

    return features, labels, stations_all, metadata_all

def order_by_weights(stations_all, features):
    """ 
    This function orderes feature 4 grid point values based on weights
    - stations_all: pandas dataframe that has column for weights
    - fetures: four dimensional numpy.ndarray where fourth dimension is parameter values in 4 closest grid cells 
    """ 
    weights_array = np.array(stations_all['weights'].str[1:-1].str.split(',',expand=True).astype(float))
    order_weights = np.flip(weights_array.argsort(axis=1), axis=1)
    luku = 0
    for i in range(0,features.shape[1]):
        features_one_station = features[:,i,:,:]
        if (luku == 0):
            features_sorted = features_one_station[:,:,order_weights[i]][:,None,:,:]
        else:
            features_sorted = np.concatenate((features_sorted, features_one_station[:,:,order_weights[i]][:,None,:,:]), axis=1)
        luku += 1
    return features_sorted

def remove_bad_stations_4(features, labels, stations_all, variable):
    """
    This function removes stations that have too little observations (checked before hand) and returns new features, labels and stations_all arrays 
    Arguments:
    - features: three dimensional numpy.ndarray
    - labels: three dimensional numpy.ndarray
    - station_features: numpy array of station features where 7th column is wmo number
    - variable: name of predicted variable, "windspeed", "windgust", "temperature", "dewpoint", "t_max" or "t_min"
    """
    removable_all = [10004,10007,10067,26425,2496,2926,6052,6017,6093,6074,2979,2818,2545,2757,2766,2827,2851,2989]
    removable_wind_and_gust = [2771,2924,2815,2722,2787,2797,2874,2767,2750,2704,2768,2829,2702,2928,2770,2828,2708,2769,2860,2830,2811,2706,2823,2798,2983,2845,2819,2756,2778,2763,2832,2880,2890,2710]

    if (variable == "windspeed"): 
        removable_stations = removable_all + removable_wind_and_gust + [6021,6044,6063,6147,2860,2935,12001]
    if (variable == "windgust"): 
        removable_stations = removable_all + removable_wind_and_gust + [10033,10037,10038,10124,10126,10136,10172,10246,10304,2044,2226,2267,2286,2293,2366,2418,2432,2435,2460,2464,2526,2550,2561,2607,2636,2670]
    if (variable == "temperature") | (variable == "dewpoint") | (variable == "t_max") | (variable == "t_min"): 
        removable_stations = removable_all + [6021,6029,6044,6063,6147,26501,6285,1018,1036,1047,1360,1368,12001,2417]
    stations_true = ~stations_all['WMON'].isin(removable_stations)
    features_new = features[:,stations_true,:,:]
    labels_new = labels[:,stations_true,:]
    stations_all_new = stations_all[stations_true]
    return features_new, labels_new, stations_all_new

def select_features_4(features, variable):
    """ 
    This function selects feature parameters that are used for that variable
    features: three dimensional numpy.ndarray where third dimension is parameters
    variable: name of predicted variable, "windspeed", "windgust", "temperature" or "dewpoint"
    """ 
    all_features_list  = ["fg","lcc","mld","p","t2m","t850","tke925","u10m","u850","u60_l","v10m","v850","v60_l","ugust10m","vgust10m", 
                          "z500","z1000","z0m","rh2m","t0m","tmax","tmin"] 
    if (variable == "windspeed"): features_list = ["fg","lcc","mld","t2m","t850","tke925","u10m","u850","u60_l","v10m","v850","v60_l","ugust10m","vgust10m","rh2m","t0m"]
    if (variable == "windgust"): features_list = ["fg","lcc","mld","t2m","t850","tke925","u10m","u850","u60_l","v10m","v850","v60_l","ugust10m","vgust10m","rh2m","t0m"]
    if (variable == "temperature"): features_list = ["fg","lcc","mld","p","t2m","t850","tke925","u850","v850","z500","z1000","rh2m","t0m","tmax","tmin"]
    if (variable == "dewpoint"): features_list = ["fg","lcc","mld","p","t2m","t850","tke925","u850","v850","z500","z1000","rh2m","t0m","tmax","tmin"]
    if (variable == "t_max"): features_list = ["fg","lcc","mld","t2m","t850","u850","v850","z500","rh2m","t0m","tmax","tmin"]
    if (variable == "t_min"): features_list = ["fg","lcc","mld","t2m","t850","u10m","v10m","z500","rh2m","t0m","tmax","tmin"]
    ilocs = [all_features_list.index(feature) for feature in features_list]
    features_new = features[:,:,ilocs,:]  
    return features_new, features_list

def add_t_inv_features_4(features, features_list):
    #Add new features to features array t850-t2m, t2m-t0m, t850-t0m
    features_list_new = features_list.copy()
    t2m_iloc = features_list.index("t2m")
    t850_iloc = features_list.index("t850")
    t0m_iloc = features_list.index("t0m")
    t850_t2m_features = features[:,:,t850_iloc,:] - features[:,:,t2m_iloc,:]
    t2m_t0m_features = features[:,:,t2m_iloc,:] - features[:,:,t0m_iloc,:]
    t850_t0m_features = features[:,:,t850_iloc,:] - features[:,:,t0m_iloc,:]
    features_new = np.concatenate([features, t850_t2m_features[:,:,np.newaxis,:], t2m_t0m_features[:,:,np.newaxis,:], t850_t0m_features[:,:,np.newaxis,:]], axis=2)
    features_list_new.extend(["t850_t2m", "t2m_t0m", "t850_t0m"])
    return features_new, features_list_new

def add_error_features_4(features, features_list, labels, metadata_all, variable, prev_hours = [12,24], error_features_zero=False):
    #Add new features of forecast error at analysis time and errors from previous forecasts at lead time 3, 6, 9, 12 hours
    features_list_new = features_list.copy()
    if (variable=="windspeed"): 
        u10_iloc = features_list.index("u10m")
        v10_iloc = features_list.index("v10m")
        forecasts_point = np.sqrt(np.power(features[:,:,u10_iloc,:],2) + np.power(features[:,:,v10_iloc,:],2))
        forecast_errors = forecasts_point - labels[:,:,0,np.newaxis]
    elif (variable=="windgust"): 
        fg_iloc = features_list.index("fg")
        forecasts_point = features[:,:,fg_iloc,:]
        forecast_errors = forecasts_point - labels[:,:,2,np.newaxis]
    elif (variable in ["temperature", "t_max", "t_min"]): 
        t2m_iloc = features_list.index("t2m")
        forecasts_point = features[:,:,t2m_iloc,:]
        forecast_errors = forecasts_point - labels[:,:,3,np.newaxis]
    elif (variable=="dewpoint"):
        t2m_iloc = features_list.index("t2m")
        rh2m_iloc = features_list.index("rh2m")
        T = features[:,:,t2m_iloc,:] 
        RH = features[:,:,rh2m_iloc,:] #Obs! RH is 0...1 not in percents
        RH[RH == 0] = 0.001
        RH[RH>1] = 1
        L = 461.5
        Rw = 2.501*10**6
        forecasts_point = T/(1-(T*np.log(RH)*(L/Rw))) #Same formula than in himan calculation
        forecast_errors = forecasts_point - labels[:,:,4,np.newaxis]
    forecast_errors_0h = np.repeat(forecast_errors[0::67,:,:], 67, axis=0)
    forecast_errors_3h = np.repeat(forecast_errors[3::67,:,:], 67, axis=0)
    forecast_errors_6h = np.repeat(forecast_errors[6::67,:,:], 67, axis=0)
    forecast_errors_9h = np.repeat(forecast_errors[9::67,:,:], 67, axis=0)
    forecast_errors_12h = np.repeat(forecast_errors[12::67,:,:], 67, axis=0)
    #Create array for error features that contains nan values
    forecast_error_features = np.full((features.shape[0], features.shape[1], 1+4*len(prev_hours),4), np.nan, dtype=np.float32)
    #Add error_feature from lead time 0 to features array
    forecast_error_features[:,:,0,:] = forecast_errors_0h
    features_list_new.extend(["fe_0h"])
    #Add error_features from previous analysis times to features array
    for a_hour,i in zip(prev_hours,range(len(prev_hours))):
        a_index = int(a_hour/12*67)
        analysis_times_a_hour = metadata_all['analysistime'] + pd.to_timedelta(a_hour, unit="hour")
        #Tell the indices where analysis time +12h or +24h matches to metadata_all time
        lt_ehto_a_hour = analysis_times_a_hour.isin(metadata_all['analysistime'])
        forecast_errors_3h_a_hour = forecast_errors_3h.copy()
        forecast_errors_3h_a_hour[lt_ehto_a_hour==False,:,:] = np.nan
        forecast_error_features[a_index::,:,(i*4+1),:] = forecast_errors_3h_a_hour[:-a_index,:,:]
        forecast_errors_6h_a_hour = forecast_errors_6h.copy()
        forecast_errors_6h_a_hour[lt_ehto_a_hour==False,:,:] = np.nan
        forecast_error_features[a_index::,:,(i*4+2),:] = forecast_errors_6h_a_hour[:-a_index,:,:]
        forecast_errors_9h_a_hour = forecast_errors_9h.copy()
        forecast_errors_9h_a_hour[lt_ehto_a_hour==False,:,:] = np.nan
        forecast_error_features[a_index::,:,(i*4+3),:] = forecast_errors_9h_a_hour[:-a_index,:,:]
        forecast_errors_12h_a_hour = forecast_errors_12h.copy()
        forecast_errors_12h_a_hour[lt_ehto_a_hour==False,:,:] = np.nan
        forecast_error_features[a_index::,:,(i*4+4),:] = forecast_errors_12h_a_hour[:-a_index,:,:]
        features_list_new.extend([f"fe_a{a_hour}_3h",f"fe_a{a_hour}_6h",f"fe_a{a_hour}_9h",f"fe_a{a_hour}_12h"])
    if error_features_zero:
        #Change all error features to zero, this is only for testing the model quality when observations are missing and error features can not be calculated
        forecast_error_features[:] = 0.0
    features_new =  np.concatenate([features, forecast_error_features], axis=2)
    return features_new, features_list_new

def order_by_time_and_leadtime_4(metadata_all, features, labels):
    """      
    This function orders features and labels arrays based on time and leadtime (primarily) in metadata_all
    Returns ordered features, labels and metadata_all 
    Arguments:
        - features: three dimensional numpy.ndarray where first dimension is time
        - labels: three dimensional numpy.ndarray where first dimension is time
        - metadata_all, pandas dataframe that contains columns for time and leadtime
    """
    order_time = np.lexsort((metadata_all['time'],metadata_all['leadtime']))
    metadata_ordered = metadata_all.iloc[order_time]
    features_ordered = features[order_time,:,:,:]
    labels_ordered = labels[order_time,:,:]
    return features_ordered, labels_ordered, metadata_ordered

def add_time_lagged_features_4(metadata_ordered, features, features_list, n_lags): 
    """ 
    This function adds new lagged features to features array
    - metadata_ordered: pandas dataframe that contains columns for (ordered) time and leadtime
    - fetures: three dimensional numpy.ndarray which is ordered based on time and leadtime
    - features_list: list of feature names (forecast_error features are not lagged)
    - n_lags, int, number of lags
    
    Returns: 
    - features_all_t: contains also lagged features
    - lt_ehto: true-false-array that shows which rows are removed due to lagging
    """ 
    leadtime = metadata_ordered['leadtime'].to_numpy()
    #Take only index of features names that do not contain fe
    features_included = np.asarray([i for i, feature in enumerate(features_list) if "fe" not in feature], dtype=np.intp)
    lt_ehto = (leadtime >= n_lags) & (leadtime <= 66)
    parts = [features[lt_ehto]]
    for lag in range(1,n_lags+1):
        mask = (leadtime >= (n_lags - lag)) & (leadtime <= (66 - lag))
        parts.append(features[mask][:, :, features_included, :])
    features_all_t = np.concatenate(parts, axis=2)
    return features_all_t, lt_ehto


def calculate_point_forecasts_4(features, features_list, stations_all, true_rows, variable):
    """
    This function returns forecast values for selected variable
    - features: four dimensional numpy.ndarray
    - features_list: list of feature names for this variable
    - true_row: true-false-array for rows where obs has nan-values
    - variable: name of predicted variable, "windspeed", "windgust", "temperature", "dewpoint", "t_max" or "t_min"
    """
    weights_array = np.array(stations_all['weights'].str[1:-1].str.split(',',expand=True).astype(float))
    order_weights = np.flip(weights_array.argsort(axis=1), axis=1)
    weights_sorted = np.take_along_axis(weights_array, order_weights, axis=1)
    if (variable=="windspeed"): 
        u10_iloc = features_list.index("u10m")
        v10_iloc = features_list.index("v10m")
        forecasts_point_u10 = np.sum(np.multiply(features[:,:,u10_iloc,:], weights_sorted[np.newaxis,:,:]), axis=2)
        forecasts_point_v10 = np.sum(np.multiply(features[:,:,v10_iloc,:], weights_sorted[np.newaxis,:,:]), axis=2)
        forecasts_point = np.sqrt(np.power(forecasts_point_u10,2) + np.power(forecasts_point_v10,2))
    if (variable=="windgust"): 
        fg_iloc = features_list.index("fg")
        forecasts_point = np.sum(np.multiply(features[:,:,fg_iloc,:], weights_sorted[np.newaxis,:,:]), axis=2)
    if (variable=="temperature"): 
        t2m_iloc = features_list.index("t2m")
        forecasts_point = np.sum(np.multiply(features[:,:,t2m_iloc,:], weights_sorted[np.newaxis,:,:]), axis=2)
    if (variable=="dewpoint"):
        t2m_iloc = features_list.index("t2m")
        rh2m_iloc = features_list.index("rh2m")
        T = np.sum(np.multiply(features[:,:,t2m_iloc,:], weights_sorted[np.newaxis,:,:]), axis=2)
        RH = np.sum(np.multiply(features[:,:,rh2m_iloc,:], weights_sorted[np.newaxis,:,:]), axis=2)#Obs! RH is 0...1 not in percents
        RH[RH == 0] = 0.001
        RH[RH>1] = 1
        L = 461.5
        Rw = 2.501*10**6
        forecasts_point = T/(1-(T*np.log(RH)*(L/Rw))) #Same formula than in himan calculation 
    if (variable=="t_max"):
        tmax_iloc = features_list.index("tmax")
        forecasts_point = np.sum(np.multiply(features[:,:,tmax_iloc,:], weights_sorted[np.newaxis,:,:]), axis=2)
    if (variable=="t_min"):
        tmin_iloc = features_list.index("tmin")
        forecasts_point = np.sum(np.multiply(features[:,:,tmin_iloc,:], weights_sorted[np.newaxis,:,:]), axis=2)
    forecasts_point1 = forecasts_point.reshape(-1)[true_rows]
    return forecasts_point1


def combine_all_features_4(features, time_features, station_features, true_rows):
    """
    This function combines forecast features, station features and time features to one 2d-numpy array
    - features: four dimensional numpy.ndarray
    - time_features: 2d numpy array
    - station_features: 2d numpy array
    - true_row: true-false-array for rows where observations have nan-values
    """
    #Each grid_point feature is a separate feature, so reshape features array to (n_samples, n_stations, n_parameters*n_grid_points)
    features3 = features.reshape(features.shape[0],features.shape[1],features.shape[2]*features.shape[3])
    features2 = features3.reshape(features3.shape[0]*features3.shape[1],features3.shape[2])
    all_features = np.concatenate((features2, time_features, station_features[:,0:6]),axis=1)[true_rows]
    return all_features
