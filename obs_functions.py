import numpy as np
import pandas as pd
import requests

def read_smartmet_obs(variable, analysis_time):
    # read observations between analysis time - 9 hours and analysis time
    starttime = (pd.to_datetime(analysis_time, format="%Y%m%d%H") - pd.Timedelta(hours=9)).strftime("%Y%m%d%H%M")
    endtime = analysis_time + "00"
    
    trad_obs = []

    # define obs parameter names used in observation database, for ws the potential ws values are used for Finland
    if variable == "temperature":
        obs_parameter = "TA_PT1M_AVG"
    elif variable == "dewpoint":
        obs_parameter = "TD_PT1M_AVG"
    elif variable == "windspeed":
        obs_parameter = "WSP_PT10M_AVG"  # potential wind speed available for Finnish stations
    elif variable == "windgust":
        obs_parameter = "WG_PT1H_MAX"

    # conventional obs are read from two distinct smartmet server producers
    # if read fails, abort program

    for producer in ["observations_fmi", "foreign"]:
        if producer == "foreign" and variable == "windspeed":
            obs_parameter = "WS_PT10M_AVG"
        url = "http://smartmet.fmi.fi/timeseries?producer={}&tz=gmt&precision=auto&starttime={}&endtime={}&timestep=180&param=fmisid,longitude,latitude,time,{}&format=json&keyword=snwc".format(
            producer, starttime, endtime, obs_parameter
        )

        resp = requests.get(url)

        testitmp = []
        testitmp2 = []
        if resp.status_code == 200:
            testitmp2 = pd.DataFrame(resp.json())
            # test if all the retrieved observations are Nan
            testitmp2 = testitmp2.empty

        if resp.status_code != 200 or testitmp2 == True or resp.json == testitmp:
            print("Not able to connect Smartmet server for observations, forecast errors can not be calculated (we assume them to be zero)")
            #returns empty dataframe
            return pd.DataFrame()
        
        trad_obs += resp.json()

    obs = pd.DataFrame(trad_obs)
    # rename observation column if WS, otherwise WS and WSP won't work
    if variable == "windspeed":  # merge columns for WSP and WS
        obs["WSP_PT10M_AVG"] = obs["WSP_PT10M_AVG"].fillna(obs["WS_PT10M_AVG"])

    count = len(trad_obs)

    if count == 0:
        return pd.DataFrame()

    return obs


def detect_outliers_zscore(variable, analysis_time, obs_data):
    # remove outliers based on zscore with separate thresholds for upper and lower tail
    if variable == "temperature" or variable == "dewpoint":
        lower_threshold = [-7, -7, -6, -5, -5, -4, -4, -4, -5, -6, -7, -7] #[-6, -6, -5, -4, -4, -4, -4, -4, -4, -5, -6, -6]
        upper_threshold = [2.5, 2.5, 2.5, 3, 4, 5, 5, 5, 3, 2.5, 2.5, 2.5]
    elif variable == "windspeed" or variable == "windgust":
        upper_threshold = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
        lower_threshold = [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4]

    thres_month = (pd.to_datetime(analysis_time, format="%Y%m%d%H")).month
    up_thres = upper_threshold[thres_month - 1]
    low_thres = lower_threshold[thres_month - 1]

    outliers = []

    #For unique analysis times
    atimes = obs_data['time'].unique()
    for atime in atimes:
        tmpobs = obs_data.iloc[:,4][obs_data['time'] == atime]
        mean = np.mean(tmpobs)
        std = np.std(tmpobs)
        value_low_th = low_thres*std - mean
        value_up_th = up_thres*std + mean
        #If value is over/under thresholds set value to NaN
        outlier_locs = obs_data.index[(obs_data['time'] == atime) & ((obs_data.iloc[:,4] < value_low_th) | (obs_data.iloc[:,4] > value_up_th))].tolist()
        outliers.extend(outlier_locs) 
    
    #Replace outliers with NaN values
    dataout = obs_data.copy()
    dataout.iloc[outliers,4] = np.nan
    # print(obs_data[obs_data.iloc[:,5].isin(outliers)])
    return outliers, dataout


def read_obs(variable, analysis_time, stations_list):
    """Read observations from smartmet server"""

    # read observations for "analysis" time == leadtime 1
    # obstime = fcstime[1]

    obs = read_smartmet_obs(variable, analysis_time)

    #Check how many fmisid are missing from observations and filter only stations in stations_list
    if not obs.empty:
        station_ids = stations_list['FMISID'].astype(str).tolist()
        missing_stations = set(station_ids) - set(obs['fmisid'].astype(str).tolist())
        print("Number of missing stations for " + variable + ": " + str(len(missing_stations)))
        obs = obs[obs['fmisid'].astype(str).isin(station_ids)].reset_index(drop=True)
        #Print min and max of observations before QC
        print("min of raw obs:", min(obs.iloc[:, 4]))
        print("max of raw obs:", max(obs.iloc[:, 4]))
        outliers, obs = detect_outliers_zscore(variable, analysis_time, obs)
        print("Number of removed outliers from observations using zscore method: " + str(len(outliers)))
        
        if not obs.empty:
            if len(outliers) > 0:
                print("min of QC obs:", min(obs.iloc[:, 4]))
                print("max of QC obs:", max(obs.iloc[:, 4]))

            #Convert temperature from Celsius to Kelvin
            if variable in ["temperature", "dewpoint", "t_max", "t_min"]:
                obs.iloc[:, 4] = obs.iloc[:, 4] + 273.15

    #Create obs table with observations at different time lags (0h, -3h, -6h, -9h) compared to analysis time
    obs_table = stations_list['FMISID'].rename('fmisid')

    #If no observations, return empty table with correct column names and NaN values
    if not obs.empty:   
        for i in ([0,3,6,9]):
            analysistime_i = (pd.to_datetime(analysis_time, format="%Y%m%d%H") - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%S")
            obs_i = obs[obs['time'] == analysistime_i][['fmisid', obs.columns[4]]].rename(columns={obs.columns[4]: f'obs_{i}h'})
            obs_table = pd.merge(obs_table, obs_i, on='fmisid', how='left')
    else: 
        for i in ([0,3,6,9]):
            #Make empty columns for different time lags
            obs_i = pd.DataFrame(np.nan, index=range(len(obs_table)), columns=[f'obs_{i}h'])
            obs_table = pd.concat([obs_table, obs_i], axis=1)
    #Print number of NaN values in obs_table
    print("Number of NaN values in obs_table for " + variable + ": " + str(obs_table.isna().sum().sum()) + "/" + str(obs_table.size-obs_table.shape[0]))
    
    return obs_table