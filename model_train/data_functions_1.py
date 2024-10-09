# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:22:33 2023

@author: ylinenk
"""

from numpy import load
import numpy as np
import pandas as pd
import math

#Define country list based on variable
def country_list(variable):
    if (variable == "windgust"):
        country_list = ['DEU','EST','FIN','LTU','LVA','NLD','SWE']
    elif (variable == "windspeed"):
        country_list = ['DEU','DNK','EST','FIN','LTU','LVA','NLD','POL','SWE']
    else:
        country_list = ['DEU','DNK', 'EST', 'FIN', 'LTU', 'LVA', 'NLD', 'NOR', 'POL', 'SWE']
    return country_list


#Load training data
def load_meps_training_data(first_year, first_month, last_year, last_month, countries, main_directory):
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

        features_country = np.empty((0, len(station_list_country), 22), dtype=np.float32)
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
                labels_part = data['lables']
                
                #Interpolate to station location using 4 grid point values
                weights1  = np.array(station_list_country['weights'].str[1:-1].str.split(',',expand=True).astype(float))
                features_part1 = np.sum(np.multiply(features_part, weights1[np.newaxis,:,np.newaxis,:]), axis=3, dtype=np.float32)

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


def remove_bad_stations(features, labels, stations_all, variable):
    """
    This function removes stations that have too little observations (checked before hand) and returns new features, labels and stations_all arrays 
    Arguments:
    - features: three dimensional numpy.ndarray
    - labels: three dimensional numpy.ndarray
    - station_features: numpy array of station features where 7th column is wmo number
    - variable: name of predicted variable, "windspeed", "windgust", "temperature", "dewpoint", "t_max" or "t_min"
    """
    if (variable == "windspeed"): removable_stations = [10004, 10007, 10067, 6021, 6044, 6063, 6147, 2860, 2935, 26425, 12001, 2496]
    if (variable == "windgust"): removable_stations = [10004,10007,10033,10037,10038,10067,10124,10126,10136,10172,10246,10304,26425,2044,2226,2267,2286,2293,2366,2418,2432,2435,2460,2464,2496,2526,2550,2561,2607,2636,2670]
    if (variable == "temperature") | (variable == "dewpoint") | (variable == "t_max") | (variable == "t_min"): 
        removable_stations = [10004, 10007, 10067, 6021, 6029, 6044, 6063, 6147, 26425, 26501, 6285, 1018, 1036, 1047, 1360, 1368, 12001, 2417, 2496]
    stations_true = ~stations_all['WMON'].isin(removable_stations)
    features_new = features[:,stations_true,:]
    labels_new = labels[:,stations_true,:]
    stations_all_new = stations_all[stations_true]
    return features_new, labels_new, stations_all_new


def copy_tmax_tmin_to_prev_hours(labels, features, metadata_all):
    """
    This function copies t_max and t_min values to 11 previous observation hours
    - labels: three dimensional numpy.ndarray where first dimension is time
    - metadata_all: pandas dataframe that contains columns for time and leadtime
    """
    #Order the metadata based on analysistime and leadtime (the data should be in that order but just to be sure)
    metadata_all['analysistime'] = pd.to_datetime(metadata_all['time'], format = '%Y-%m-%d %H:%M:%S') - pd.to_timedelta(metadata_all['leadtime'], unit="hour")
    order_analysistime = np.lexsort((metadata_all['leadtime'],metadata_all['analysistime']))

    metadata_all_order = metadata_all.iloc[order_analysistime]
    datetime_object = pd.to_datetime(metadata_all_order['time'], format = '%Y-%m-%d %H:%M:%S')
    hour = datetime_object.dt.hour
    leadtime = metadata_all_order['leadtime']

    new_labels = labels[order_analysistime]
    new_features = features[order_analysistime]
    #Get ilocs for tmax where hour is 18 and leadtime is greater than 12 and for tmin where hour is 6 and leadtime is greater than 12
    ilocs_tmax = np.where((hour == 18) & (leadtime >= 12))[0]
    ilocs_tmin = np.where((hour == 6) & (leadtime >= 12))[0]
    #Get tmax and tmin values from labels and repeat them to 11 previous hours
    tmax_values = new_labels[ilocs_tmax,:,5]
    tmin_values = new_labels[ilocs_tmin,:,6]
    #Set first all tmax and tmin values to nan
    new_labels[:,:,5:7] = np.nan
    for i in range(0,12):
        new_labels[ilocs_tmax-i,:,5] = tmax_values
        new_labels[ilocs_tmin-i,:,6] = tmin_values
    
    return new_labels, new_features, metadata_all_order


def order_by_time_and_leadtime(metadata_all, features, labels):
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
    features_ordered = features[order_time,:,:]
    labels_ordered = labels[order_time,:,:]
    return features_ordered, labels_ordered, metadata_ordered


def select_features(features, variable):
    """ 
    This function selects feature parameters that are used for that variable
    fetures: three dimensional numpy.ndarray where third dimension is parameters
    variable: name of predicted variable, "windspeed", "windgust", "temperature" or "dewpoint"
    """ 
    all_features_list  = ["fg","lcc","mld","p","t2m","t850","tke925","u10m","u850","u60_l","v10m","v850","v60_l","ugust10m","vgust10m", 
                          "z500","z1000","z0m","rh2m","t0m","tmax","tmin"] 
    if (variable == "windspeed"): features_list = ["fg","lcc","mld","tke925","u10m","u850","u60_l","v10m","v850","v60_l","ugust10m","vgust10m","rh2m","t0m"]
    if (variable == "windgust"): features_list = ["fg","lcc","mld","tke925","u10m","u850","u60_l","v10m","v850","v60_l","ugust10m","vgust10m","rh2m","t0m"]
    if (variable == "temperature"): features_list = ["fg","lcc","mld","p","t2m","t850","tke925","u850","v850","z500","z1000","rh2m","t0m","tmax","tmin"]
    if (variable == "dewpoint"): features_list = ["fg","lcc","mld","p","t2m","t850","tke925","u850","v850","z500","z1000","rh2m","t0m","tmax","tmin"]
    if (variable == "t_max"): features_list = ["fg","lcc","mld","t2m","t850","u850","v850","z500","rh2m","t0m","tmax","tmin"]
    if (variable == "t_min"): features_list = ["fg","lcc","mld","t2m","t850","u10m","v10m","z500","rh2m","t0m","tmax","tmin"]
    ilocs = [all_features_list.index(feature) for feature in features_list]
    features_new = features[:,:,ilocs]  
    return features_new, features_list


def add_time_lagged_features(metadata_ordered, features, n_lags): 
    """ 
    This function adds new lagged features to features array
    - metadata_ordered: pandas dataframe that contains columns for (ordered) time and leadtime
    - fetures: three dimensional numpy.ndarray which is ordered based on time and leadtime
    - n_lags, int, number of lags
    
    Returns: 
    - features_all_t: contains also lagged features
    - lt_ehto: true-false-array that shows which rows are removed due to lagging
    """ 
    lt_ehto = (metadata_ordered['leadtime'] >= n_lags) & (metadata_ordered['leadtime'] <= 66)
    features_all_t = features[lt_ehto]
    for i in range(1,n_lags+1):
        features_t = features[(metadata_ordered['leadtime'] >= (n_lags-i)) & (metadata_ordered['leadtime'] <= (66-i))]
        #Features must be ordered primarily based on leadtime and secondary on time 
        features_all_t = np.concatenate((features_all_t, features_t), axis=2)
    return features_all_t, lt_ehto


def create_time_features(metadata_ordered, n_stations, lt_ehto, leadtime = True):
    """
    This function creates time features (month_sin, month_cos, hour_sin, hour_cos and leadtime) using time
    from metadata_ordered
    - metadata_ordered: pandas dataframe that contains columns for (ordered) time and leadtime
    - n_stations: int, number of stations in features array (features.shape[1])
    - lt_ehto: true-false-array that shows which rows have to be removed due to lagging
    """
    datetime_object = pd.to_datetime(metadata_ordered['time'], format = '%Y-%m-%d %H:%M:%S')
    features_time = pd.DataFrame({
        'month_sin': np.sin(datetime_object.dt.month*2*math.pi/12),
        'month_cos': np.cos(datetime_object.dt.month*2*math.pi/12),
        'hour_sin': np.sin(datetime_object.dt.hour*2*math.pi/23),
        'hour_cos': np.cos(datetime_object.dt.hour*2*math.pi/23),
        'leadtime': np.array(metadata_ordered['leadtime'])})
    if leadtime == False: features_time = features_time.drop(columns=['leadtime'])
    time_features = np.repeat(np.array(features_time, dtype=np.float32)[lt_ehto],n_stations,axis=0)
    return time_features


def create_station_features(stations_all, n_times):
    """
    This function creates station features (lat, lon, elev, lsm, z0, elev_D, WMO) from stations_all dataframe
    - stations_all: pandas dataframe that contains columns for LAT, LON, ELEV, lsm, Z0, and WMON
    - n_times: int, number of times in features array (features.shape[0])
    """
    features_station = pd.DataFrame({
        'lat': stations_all['LAT'],
        'lon': stations_all['LON'],
        'elev': stations_all['ELEV'],
        'lsm': stations_all['lsm'],
        'z0': stations_all['Z0'],
        'elev_D': stations_all['ELEV']-stations_all['Z0']/9.80655,
        'WMO': stations_all['WMON']})
    station_features = np.repeat(np.array(features_station, dtype=np.float32)[None,:,:],n_times, axis=0).reshape(-1,7)
    return station_features


def select_observation_variable(labels, lt_ehto, variable):
    """ 
    This function selects given variable data from labels array
    - labels: three dimensional numpy.ndarray where third dimension is variable
    - lt_ehto: true-false-array that shows which rows have to be removed due to lagging
    - variable: name of predicted variable, "windspeed", "windgust", "temperature" or "dewpoint"
    """ 
    if (variable == "windspeed"): observations = labels[lt_ehto,:,0].reshape(-1)
    if (variable == "winddirection"): observations = labels[lt_ehto,:,1].reshape(-1)
    if (variable == "windgust"): observations = labels[lt_ehto,:,2].reshape(-1)
    if (variable == "temperature"): observations = labels[lt_ehto,:,3].reshape(-1)
    if (variable == "dewpoint"): observations = labels[lt_ehto,:,4].reshape(-1)
    if (variable == "t_max"): observations = labels[lt_ehto,:,5].reshape(-1)
    if (variable == "t_min"): observations = labels[lt_ehto,:,6].reshape(-1)
    return observations


def rows_with_good_observations(observations, variable):
    """
    This function gives true-false-array for rows where nan-values exist and where observations are not too high or too low
    - observations: numpy array of observations 
    - variable: name of predicted variable, "windspeed", "windgust", "temperature" or "dewpoint"
    """
    if (variable == "windspeed"): true_rows = ~np.isnan(observations) & ~(observations>45) 
    if (variable == "windgust"): true_rows = ~np.isnan(observations) & ~(observations>55)
    if (variable == "temperature"): true_rows = ~np.isnan(observations) & ~(observations < 220) & ~(observations > 320)
    if (variable == "dewpoint"): true_rows = ~np.isnan(observations) & ~(observations < 210) & ~(observations > 310)
    if (variable == "t_max"): true_rows = ~np.isnan(observations) & ~(observations < 223) & ~(observations > 323)
    if (variable == "t_min"): true_rows = ~np.isnan(observations) & ~(observations < 218) & ~(observations > 320)
    return true_rows 


def calculate_point_forecasts(features, features_list, true_rows, variable):
    """
    This function returns forecast values for selected variable
    - features: three dimensional numpy.ndarray
    - features_list: list of feature names for this variable
    - true_row: true-false-array for rows where obs has nan-values
    - variable: name of predicted variable, "windspeed", "windgust", "temperature", "dewpoint", "t_max" or "t_min"
    """
    features2 = features.reshape(features.shape[0]*features.shape[1],features.shape[2])
    if (variable=="windspeed"): 
        u10_iloc = features_list.index("u10m")
        v10_iloc = features_list.index("v10m")
        forecasts_point = np.sqrt(np.power(features2[:,u10_iloc][true_rows],2) + np.power(features2[:,v10_iloc][true_rows],2))
    if (variable=="windgust"): 
        fg_iloc = features_list.index("fg")
        forecasts_point = features2[:,fg_iloc][true_rows] 
    if (variable=="temperature"): 
        t2m_iloc = features_list.index("t2m")
        forecasts_point = features2[:,t2m_iloc][true_rows]
    if (variable=="dewpoint"):
        t2m_iloc = features_list.index("t2m")
        rh2m_iloc = features_list.index("rh2m")
        T = features2[:,t2m_iloc][true_rows]
        RH = features2[:,rh2m_iloc][true_rows] #Obs! RH is 0...1 not in percents
        RH[RH == 0] = 0.001
        RH[RH>1] = 1
        L = 461.5
        Rw = 2.501*10**6
        forecasts_point = T/(1-(T*np.log(RH)*(L/Rw))) #Same formula than in himan calculation 
    if (variable=="t_max"):
        tmax_iloc = features_list.index("tmax")
        forecasts_point = features2[:,tmax_iloc][true_rows]
    if (variable=="t_min"):
        tmin_iloc = features_list.index("tmin")
        forecasts_point = features2[:,tmin_iloc][true_rows]
    return forecasts_point


def combine_all_features(features, time_features, station_features, true_rows):
    """
    This function combines forecast features, station features and time features to one 2d-numpy array
    - features: three dimensional numpy.ndarray
    - time_features: 2d numpy array
    - station_features: 2d numpy array
    - true_row: true-false-array for rows where nan-values and bad stations are false
    """
    features2 = features.reshape(features.shape[0]*features.shape[1],features.shape[2])
    all_features = np.concatenate((features2, time_features, station_features[:,0:6]),axis=1)[true_rows]
    return all_features
