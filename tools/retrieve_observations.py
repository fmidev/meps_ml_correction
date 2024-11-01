#This script retrieves observations from the smartmet server for the training data. Run this script in the folder where the trainingdata folder is located
#Run: python3 retrieve_observations_2.py {first_year} {first_month} {last_year} {last_month} (> retrieve_observations.log)
#Eg. python3 retrieve_observations_2.py 2024 1 2024 12 (> retrieve_observations_2024.log)

import os
import numpy as np
from numpy import load
import pandas as pd
import requests
import io
import time
import datetime
import sys

'''
set these parameters
'''
countries = ['FIN','NLD','NOR','SWE','DEU','DNK','EST','LVA','LTU','POL']
params_fin = ["time","WSP_PT10M_AVG","WD_PT10M_AVG","WG_PT1H_MAX","TA_PT1M_AVG","TD_PT1M_AVG","TA_PT12H_MAX","TA_PT12H_MIN"]
params_other = ["time","WS_PT10M_AVG","WD_PT10M_AVG","WG_PT1H_MAX","TA_PT1M_AVG","TD_PT1M_AVG","TA_PT12H_MAX","TA_PT12H_MIN"]

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

first_year=int(sys.argv[1])
first_month=int(sys.argv[2])
last_year=int(sys.argv[3])
last_month=int(sys.argv[4])

def obs_from_smartmet_try(start, end, step, params, location, producer='foreign'):
    """
    returns time series of observations from smartmet server
    """
    proxies = {'http':os.getenv('https_proxy')}
    timezone = 'gmt'
    url = 'http://smartmet.fmi.fi/timeseries?format=ascii&timeformat=sql&starttime={start}&endtime={end}&timestep={step}h&fmisid={sid}&producer={prod}&tz={tz}&precision=double&param={params}&separator=;'.format(start=start,end=end,step=str(step),prod=producer,tz=timezone,params=','.join(params),sid=str(location))
    count=0
    while (count<10):
        try:
            response = requests.get(url, proxies=proxies, timeout=5).text
            return pd.read_csv(io.StringIO(response), names=params, sep=';')
        except requests.exceptions.Timeout:
            if (count<5):
                #print(f"Request timed out after {5} seconds. Wait 5 seconds. ({count})")
                time.sleep(5)
            else:
                print(location)
                print(f"Request timed out after {5} seconds. Wait 15 seconds. ({count})")
                time.sleep(15)
            count+=1
            continue
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
    return None

for maa in countries:
    print(maa)
    station_list = pd.read_csv('trainingdata/' + maa + '/' + 'stations.csv') 

    if (maa == 'FIN'):
        params = params_fin
    else:
        params = params_other
               
    for yyyy in list(range(first_year,last_year+1)):    
        for mm in [1,2,3,4,5,6,7,8,9,10,11,12]:
            if ((yyyy == first_year) & (mm < first_month)): continue
            if ((yyyy == last_year) & (mm > last_month)): break
            print("year/month: " + str(yyyy) + "/" + str(mm))
            
            data = load('trainingdata/' + maa + '/' + str(yyyy) + '/' + str(mm) + '/trainingdata.npz')
            metadata = pd.read_csv('trainingdata/' + maa + '/' + str(yyyy) + '/' + str(mm) + '/metadata.csv')
            
            features = data['features']
            
            start_time = min(metadata['time'])
            end_time = max(metadata['time'])
            
            labels_new = np.empty((len(metadata), 0, len(params)-1), dtype=np.float64)
            i = 0
            
            for fmisid in station_list['FMISID']:
                #print(fmisid)
                if (maa == 'FIN'):
                    havainnot = obs_from_smartmet_try(start_time,end_time,1,params,fmisid,producer="observations_fmi")
                else:
                    havainnot = obs_from_smartmet_try(start_time,end_time,1,params,fmisid)

                if (len(havainnot) == 0):
                    merged_array = np.full([len(metadata),len(params)-1], np.nan, dtype=np.float64)
                else:
                    #Convert temperature from Celsius to Kelvin 
                    havainnot['TA_PT1M_AVG'] = havainnot['TA_PT1M_AVG'] + 273.15
                    havainnot['TD_PT1M_AVG'] = havainnot['TD_PT1M_AVG'] + 273.15
                    havainnot['TA_PT12H_MAX'] = havainnot['TA_PT12H_MAX'] + 273.15
                    havainnot['TA_PT12H_MIN'] = havainnot['TA_PT12H_MIN'] + 273.15
                    merged_havainnot = pd.merge(right=metadata,left=havainnot,sort=False,how="right")
                    merged_array = np.array(merged_havainnot.loc[:,params[1:]], dtype=np.float64)

                labels_new = np.concatenate((labels_new, merged_array[:,None,:]), axis=1, dtype=np.float64)

                #Print station id if more than half of the observations are missing
                if (sum(np.isnan(merged_array[:,0])) > (metadata.shape[0]/2)):
                    print("WS, wmon:",station_list['WMON'][i],", nan amount:",sum(np.isnan(merged_array[:,0])))
                if (sum(np.isnan(merged_array[:,2])) > (metadata.shape[0]/2)):
                    print("WG, wmon:",station_list['WMON'][i],", nan amount:",sum(np.isnan(merged_array[:,2])))
                if (sum(np.isnan(merged_array[:,3])) > (metadata.shape[0]/2)):
                    print("T2m, wmon:",station_list['WMON'][i],", nan amount:",sum(np.isnan(merged_array[:,3])))
                i += 1
            np.savez('trainingdata/' + maa + '/' + str(yyyy) + '/' + str(mm) + '/trainingdata.npz', features = features, lables = labels_new)

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("End!")
