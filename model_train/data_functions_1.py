# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:22:33 2023

@author: ylinenk
"""

from numpy import load
import numpy as np
import pandas as pd
import math

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
        #os.chdir('/data/statcal/projects/MEPS_WS_correction/trainingdata/' + maa)
        maa_dir = (main_directory + maa)
        station_list_country = pd.read_csv(maa_dir + '/stations2.csv') #Z0 ja lsm lisätty tähän asemalistaan
        stations_list.append(station_list_country)

        features_country = np.empty((0, len(station_list_country), 20), dtype=np.float32)
        labels_country = np.empty((0, len(station_list_country), 4), dtype=np.float32)

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

                features_country = np.concatenate((features_country, features_part1[:,:,0:20]), axis=0, dtype=np.float32)
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


#FUNKTIO: order_by_time_and_leadtime(metadata_all, features, labels), return features, labels, metadata_ordered
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
    variable: name of predicted variable, "windspeed", "windgust" or "temperature"
    """ 
    #fg, lcc, mld, p, t2m, t850, tke925, u10m, u850, u60_l, v10m, v850, v60_l, ugust10m, vgust10m, z500, z1000, z0m, rh2m, t0m 
    if (variable == "windspeed"): features = features[:,:,0:17] 
    if (variable == "windgust"): features = features[:,:,0:17]
    if (variable == "temperature"): features = features[:,:,[0,1,2,3,4,5,6,8,11,15,16,18,19]]
    return features


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
        #Features must be ordered primarily based on leadtimen and secondary on time 
        features_all_t = np.concatenate((features_all_t, features_t), axis=2)
    return features_all_t, lt_ehto


def create_time_features(metadata_ordered, n_stations, lt_ehto):
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
    - variable: name of predicted variable, "windspeed", "windgust" or "temperature"
    """ 
    if (variable == "windspeed"): observations = labels[lt_ehto,:,0].reshape(-1)
    if (variable == "winddirection"): observations = labels[lt_ehto,:,1].reshape(-1)
    if (variable == "windgust"): observations = labels[lt_ehto,:,2].reshape(-1)
    if (variable == "temperature"): observations = labels[lt_ehto,:,3].reshape(-1)
    return observations


def rows_with_good_observations(observations, station_features, variable):
    """
    This function gives true-false-array for rows where nan-values and bad stations are false
    - observations: numpy array of observations 
    - station_features: numpy array of station features where 7th column is wmo number
    - variable: name of predicted variable, "windspeed", "windgust" or "temperature"
    """
    if (variable == "windspeed"):
        removable_stations = [10067,6021,6108,6044,6033,6063,1018,1368,1360,1047,12001,2496]
        remove_station_rows = np.array([value in removable_stations for value in station_features[:,6]])
        #Remove nan-observations (obs qc must be done beforehand)
        true_rows = ~np.isnan(observations) & ~remove_station_rows & ~(observations>45) 
    if (variable == "windgust"):
        removable_stations = [10004,10007,10136,10126,10246,10037,10038,10067,10033,10172,10124,10304,1483,1360,1047,1384,2496,2526,2607,2636,2670,2226,2267,2460,2418,2286,2550,2464,2435,2432,2044,2293,2366]
        #POISTA myös kaikki DNK, LTU ja POL
        remove_station_rows = np.array([value in removable_stations for value in station_features[:,6]])
        #Remove nan-observations (obs qc must be done beforehand)
        true_rows = ~np.isnan(observations) & ~remove_station_rows & ~(observations>55)
    if (variable == "temperature"):
        removable_stations = [10004,10007]
        remove_station_rows = np.array([value in removable_stations for value in station_features[:,6]])
        true_rows = ~np.isnan(observations) & ~remove_station_rows & ~(observations > 320) & ~(observations < 220)
    return true_rows 


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


def calculate_point_forecasts(features, true_rows, variable):
    """
    This function returns forecast values for selected variable
    - stations_all: pandas dataframe that contains column for weights
    - features: three dimensional numpy.ndarray
    - true_row: true-false-array for rows where nan-values and bad stations are false
    - variable: name of predicted variable, "windspeed", "windgust" or "temperature"
    """
    features2 = features.reshape(features.shape[0]*features.shape[1],features.shape[2])
    if (variable=="windspeed"): forecasts_point = np.sqrt(np.power(features2[:,7][true_rows],2) + np.power(features2[:,10][true_rows],2)) 
    if (variable=="windgust"): forecasts_point = features2[:,0][true_rows] 
    if (variable=="temperature"): forecasts_point = features2[:,4][true_rows] 
    return forecasts_point


