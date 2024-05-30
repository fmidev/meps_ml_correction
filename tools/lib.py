import eccodes as ecc
import pandas as pd
import json
import requests
import io
import os

"""
A collection of useful functions to build support tools for ML based wind speed corrections
"""

"""
Arcus parameter description class for grib2 keys
param = [discipline,category,number]
level = [levelType,levelValue]
mbr = member
"""
class param:
    def __init__(self, param, level, mbr, statistical_process = 'NULL'):
        self.param = param
        self.level = level
        self.mbr = mbr
        self.statistical_process = statistical_process

"""
return an index for given date and analysis time
"""
def get_index(s3,date):
    key = 'MEPS_prod/{}/index.json'.format(date.strftime('%Y/%m/%d/%H'))
    f = s3.Object('calibration',key).get()['Body']

    return json.loads(f.read())

"""
return handle to a grib message in memory from arcus archive
"""
def grib_message_from_arcus(s3,bucket,key,offset,length):
    f = s3.Object(bucket,key).get(Range="bytes={}-{}".format(offset,offset+length-1))['Body']
    gid = ecc.codes_new_from_message(f.read())
    return gid

"""
returns time series of observations for parameters in params at a location as pandas table from smartmet server
"""
def obs_from_smartmet(start, end, step, params, location, producer='observations_fmi',levels='0'):
    proxies = {'http':os.getenv('https_proxy')}

    timezone = 'gmt'
    url = 'http://smartmet.fmi.fi/timeseries?format=ascii&starttime={start}&endtime={end}&timestep={step}h&wmo={sid}&producer={prod}&tz={tz}&precision=double&param={params}&levels={lvl}&separator=;'.format(start=start.strftime('%Y%m%d%H%M%S'),end=end.strftime('%Y%m%d%H%M%S'),step=step,prod=producer,tz=timezone,params=','.join(params),sid=location,lvl=levels)
    response = requests.get(url, proxies=proxies).text
    return pd.read_csv(io.StringIO(response), names=params, sep=';')
