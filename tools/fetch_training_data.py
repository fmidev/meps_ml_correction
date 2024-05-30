import os,sys
import boto3
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from ast import literal_eval
import csv
import getopt
from pathlib import Path

from lib import *

"""
set these global parameters, e.g. t2m control member
"""
fg = param([0,2,22],[103,10],0,2)
mld = param([0,19,3],[103,0],0)
p = param([0,3,0],[102,0],0)
t2 = param([0,0,0],[103,2],0)
t850hPa = param([0,0,0],[100,850],0)
tke = param([0,19,11],[100,925],0)
u10 = param([0,2,2],[103,10],0)
u850 = param([0,2,2],[100,850],0)
u65 = param([0,2,2],[105,65],0)
v10 = param([0,2,3],[103,10],0)
v850 = param([0,2,3],[100,850],0)
v65 = param([0,2,3],[105,65],0)
ugust = param([0,2,23],[103,10],0,2)
vgust = param([0,2,24],[103,10],0,2)
z500 = param([0,3,4],[100,500],0)
z1000 = param([0,3,4],[100,1000],0)
z0 = param([0,3,4],[103,0],0)
lcc = param([0,6,194],[103,0],0)
r2 = param([0,1,192],[103,2],0)
t0 = param([0,0,0],[103,0],0)
par = [fg,lcc,mld,p,t2,t850hPa,tke,u10,u850,u65,v10,v850,v65,ugust,vgust,z500,z1000,z0,r2,t0]

countries = ['FIN','NLD','NOR','SWE','DEU','DNK','EST','LVA','LTU','POL']

"""
define a date generator
"""
def generate_date(begin,end,step):
    while(begin != end):
        yield begin
        begin += step

"""
make a nearest point list for stations
"""
def make_points(stationfile):
    df = pd.read_csv(stationfile)
    df['nearest'] = df['nearest'].apply(literal_eval)
    y = dict()
    for country in countries:
        y[country] = []
        for x in df[df['Country'] == country]['nearest']:
            y[country].extend(x)
    return y

"""
create a generator to provide tensor flow input data from arcus and smartmet
"""
def generate_grib(start,end,step,points,stations,station_list):
    date = datetime.datetime.strptime(start,'%Y-%m-%dT%H:%M:%S')
    end = datetime.datetime.strptime(end,'%Y-%m-%dT%H:%M:%S')
    step = datetime.timedelta(hours=int(step))
    s3 = boto3.resource(
    's3',
    endpoint_url=os.getenv('S3_HOSTNAME'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    
    while(date != end):
        index = get_index(s3,date)
        obs = pd.concat([obs_from_smartmet(date,date + datetime.timedelta(hours=66),1,['wmo','time','WSP_PT10M_AVG','WD_PT10M_AVG','WG_PT1H_MAX','TA_PT1M_AVG','TD_PT1M_AVG'],stations['FIN']).rename(columns={"WSP_PT10M_AVG": "windspeed", "WD_PT10M_AVG": "winddirection", "WG_PT1H_MAX": "windgust", "TA_PT1M_AVG": "temperature", "TD_PT1M_AVG": "dewpoint"}),
                        pd.concat([obs_from_smartmet(date,date + datetime.timedelta(hours=66),1,['wmo','time','WS_PT10M_AVG','WD_PT10M_AVG','WG_PT1H_MAX','TA_PT1M_AVG','TD_PT1M_AVG'],s,'foreign').rename(columns={"WS_PT10M_AVG": "windspeed", "WD_PT10M_AVG": "winddirection", "WG_PT1H_MAX": "windgust", "TA_PT1M_AVG": "temperature", "TD_PT1M_AVG": "dewpoint"}) for k,s in stations.items()])])
        for leadtime in range(67):
            print(date)
            mydate = date + datetime.timedelta(hours=leadtime)
            x = dict()
            [x.update({country : np.zeros([1,len(par),len(points[country])//4,4],dtype=float)}) for country in stations.keys()]
            n = 0
            
            for p in par:
                idx = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(2,leadtime,p.level[0],p.level[1],p.mbr,p.param[0],p.param[1],p.param[2],p.statistical_process)
                key, offset, length = tuple(index[idx])
                gid = grib_message_from_arcus(s3,"calibration",key,offset,length)
                for country in countries:
                    v = np.array(ecc.codes_get_values(gid))[points[country]].reshape([len(points[country])//4,4])
                    x[country][0,n,:,:] = v
                ecc.codes_release(gid)
                n += 1

            for country in countries:
                y = pd.merge(station_list[station_list['Country'] == country],obs.loc[obs['time'] == mydate.strftime('%Y%m%dT%H%M%S')][["wmo","windspeed","winddirection","windgust","temperature","dewpoint"]].rename(columns={"wmo" : "WMON"}),how='left',on='WMON')[["windspeed","winddirection","windgust","temperature","dewpoint"]].to_numpy()
                yield np.transpose(x[country],[0,2,1,3]),y.reshape([1,len(points[country])//4,5]),[leadtime,mydate]
            
        date += step

def main():
    """
    Fetch command line arguments
    """
    #------------------------------------------------------------------------------------------------------------
    options, remainder = getopt.getopt(sys.argv[1:],[],['year=','month=','stationfile=','help'])

    for opt, arg in options:
        if opt == '--help':
            print('*.py --year <YYYY> --month <MM> --stationfile <filename>')
            exit()
        elif opt == '--year':
                year = int(arg)
        elif opt == '--month':
                month = int(arg)
        elif opt == '--stationfile':
                stationfile = arg

    # Exit with error message if not all command line arguments are specified
    try:
        year, month, stationfile
    except NameError:
        print('ERROR! Not all input parameters specified: *.py --year <YYYY> --month <MM> --stationfile <filename>')
        exit()
    #------------------------------------------------------------------------------------------------------------

    """
    create a dictionary of station lists by country
    """
    #------------------------------------------------------------------------------------------------------------
    stations = dict()

    station_list=pd.read_csv(stationfile)
    for Country in countries:
        stations[Country] = ','.join(map(str,station_list[station_list['Country'] == Country]['WMON'].values.tolist()))
    #------------------------------------------------------------------------------------------------------------

    """
    Set time period, create input data to one month chunks of data
    Array start,end and month need to be same length
    """
    #------------------------------------------------------------------------------------------------------------
    startdate = datetime.datetime(year,month,1)
    enddate = startdate + relativedelta(months=1)

    f = generate_grib(startdate.strftime('%Y-%m-%dT%H:%M:%S'),enddate.strftime('%Y-%m-%dT%H:%M:%S'),12,make_points(stationfile),stations,station_list)
    
    A = dict()
    B = dict()

    time = dict()

    for country in countries:
        A[country],B[country],T = next(f)
        time[country] = [T]

    flag = True
    while flag:
        for country in countries:
            try:
                X,Y,T = next(f)
            except:
                flag = False
                break 
            A[country] = np.append(A[country],X,axis=0)
            B[country] = np.append(B[country],Y,axis=0)
            time[country].append(T)

    for country in countries:
        Path(f'trainingdata/{country}/{year}/{month}').mkdir(parents=True, exist_ok=True)
        np.savez(f'trainingdata/{country}/{year}/{month}/trainingdata.npz',features=A[country],lables=B[country])


    for country in countries:       
        with open(f'trainingdata/{country}/{year}/{month}/metadata.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(['leadtime','time'])
            write.writerows(time[country])
    #------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
