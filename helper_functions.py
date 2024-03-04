# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:15:33 2023

@author: ylinenk
"""

import os
import eccodes as ecc
import gridpp
import numpy as np
import eccodes as ecc
import sys
import pyproj
import datetime
import pandas as pd
import math
import xgboost as xgb
import quantile_mapping as qm
import fsspec
import s3fs
from multiprocessing import Process, Queue

def get_shapeofearth(gh):
    """Return correct shape of earth sphere / ellipsoid in proj string format.
    Source data is grib2 definition.
    """

    shape = ecc.codes_get_long(gh, "shapeOfTheEarth")
    
    if shape == 1:
        v = ecc.codes_get_long(gh, "scaledValueOfRadiusOfSphericalEarth")
        s = ecc.codes_get_long(gh, "scaleFactorOfRadiusOfSphericalEarth")
        return "+R={}".format(v * pow(10, s))
    
    if shape == 5:
        return "+ellps=WGS84"
    
    if shape == 6:
        return "+R=6371229.0"


def get_falsings(projstr, lon0, lat0):
    """Get east and north falsing for projected grib data"""
    
    ll_to_projected = pyproj.Transformer.from_crs("epsg:4326", projstr)
    return ll_to_projected.transform(lat0, lon0)


def get_projstr(gh):
    """Create proj4 type projection string from grib metadata" """
    
    projstr = None
    
    proj = ecc.codes_get_string(gh, "gridType")
    first_lat = ecc.codes_get_double(gh, "latitudeOfFirstGridPointInDegrees")
    first_lon = ecc.codes_get_double(gh, "longitudeOfFirstGridPointInDegrees")
    
    if proj == "polar_stereographic":
        projstr = "+proj=stere +lat_0=90 +lat_ts={} +lon_0={} {} +no_defs".format(
            ecc.codes_get_double(gh, "LaDInDegrees"),
            ecc.codes_get_double(gh, "orientationOfTheGridInDegrees"),
            get_shapeofearth(gh),
        )
        fe, fn = get_falsings(projstr, first_lon, first_lat)
        projstr += " +x_0={} +y_0={}".format(-fe, -fn)
        
    elif proj == "lambert":
        projstr = (
            "+proj=lcc +lat_0={} +lat_1={} +lat_2={} +lon_0={} {} +no_defs".format(
                ecc.codes_get_double(gh, "Latin1InDegrees"),
                ecc.codes_get_double(gh, "Latin1InDegrees"),
                ecc.codes_get_double(gh, "Latin2InDegrees"),
                ecc.codes_get_double(gh, "LoVInDegrees"),
                get_shapeofearth(gh),
            )
        )
        fe, fn = get_falsings(projstr, first_lon, first_lat)
        projstr += " +x_0={} +y_0={}".format(-fe, -fn)
        
    else:
        print("Unsupported projection: {}".format(proj))
        sys.exit(1)
        
    return projstr


def read_file_from_s3(grib_file):
    uri = "simplecache::{}".format(grib_file)
    
    return fsspec.open_local(
        uri,
        mode="rb",
        s3={"anon": True, "client_kwargs": {"endpoint_url": "https://lake.fmi.fi"}},
    )


def read_grib(gribfile, read_coordinates=False):
    """Read first message from grib file and return content.
    List of coordinates is only returned on request, as it's quite
    slow to generate.
    """
    forecasttime = []
    leadtime = []
    values = []
    
    print(f"Reading file {gribfile}")
    wrk_gribfile = gribfile
    
    if gribfile.startswith("s3://"):
        wrk_gribfile = read_file_from_s3(gribfile)
        
    lons = []
    lats = []
    
    with open(wrk_gribfile, "rb") as fp:
        # print("Reading {}".format(gribfile))
        
        while True:
            try:
                gh = ecc.codes_grib_new_from_file(fp)
            except ecc.WrongLengthError as e:
                print(e)
                file_stats = os.stat(wrk_gribfile)
                print("Size of {}: {}".format(wrk_gribfile, file_stats.st_size))
                sys.exit(1)
            
            if gh is None:
                break
            
            ni = ecc.codes_get_long(gh, "Nx")
            nj = ecc.codes_get_long(gh, "Ny")
            dataDate = ecc.codes_get_long(gh, "dataDate")
            dataTime = ecc.codes_get_long(gh, "dataTime")
            forecastTime = ecc.codes_get_long(gh, "endStep")
            leadtime.append(forecastTime)
            analysistime = datetime.datetime.strptime(
                "{}.{:04d}".format(dataDate, dataTime), "%Y%m%d.%H%M"
            )
            
            ftime = analysistime + datetime.timedelta(hours=forecastTime)
            forecasttime.append(ftime)
            
            tempvals = ecc.codes_get_values(gh).reshape(nj, ni)
            values.append(tempvals)

            if read_coordinates and len(lons) == 0:
                projstr = get_projstr(gh)
                
                di = ecc.codes_get_double(gh, "DxInMetres")
                dj = ecc.codes_get_double(gh, "DyInMetres")
                
                proj_to_ll = pyproj.Transformer.from_crs(projstr, "epsg:4326")
                
                for j in range(nj):
                    y = j * dj
                    for i in range(ni):
                        x = i * di
                        
                        lat, lon = proj_to_ll.transform(x, y)
                        lons.append(lon)
                        lats.append(lat)
            ecc.codes_release(gh)
        

        #Order values, leadtime, forecasttime based on leadtime
        values = [values[i] for i in np.array(leadtime)]
        leadtime.sort()
        forecasttime.sort()
                                                        
        if read_coordinates == False and len(values) == 1:
            return (
                None,
                None,
                np.asarray(values).reshape(nj, ni),
                leadtime,
                analysistime,
                forecasttime,
            )
        elif read_coordinates == False and len(values) > 1:
            return None, None, np.asarray(values), leadtime, analysistime, forecasttime
        else:
            return (
                np.asarray(lons).reshape(nj, ni),
                np.asarray(lats).reshape(nj, ni),
                np.asarray(values),
                leadtime,
                analysistime,
                forecasttime,
            )


def read_grid(args):
    """Top function to read "all" gridded data"""
    # Define the grib-file used as background/"parameter_data"
    if args.parameter == "temperature":
        lons, lats, vals, leadtime, analysistime, forecasttime = read_grib(args.t2_data, True)
    elif args.parameter == "windspeed":
        lons, lats, vals_u, leadtime, analysistime, forecasttime = read_grib(args.u10_data, True)
        _, _, vals_v, _, _, _ = read_grib(args.v10_data, True)
        vals = np.sqrt(np.power(vals_u,2) + np.power(vals_v,2))
    elif args.parameter == "windgust":
        lons, lats, vals, leadtime, analysistime, forecasttime = read_grib(args.fg_data, True)

    _, _, topo, _, _, _ = read_grib(args.topography_data, False)
    _, _, lc, _, _, _ = read_grib(args.landseacover_data, False)

    # modify  geopotential to height and use just the first grib message, since the topo & lc fields are static
    topo = topo / 9.81
    if(len(topo.shape) == 3): topo = topo[0]
    if(len(lc.shape) == 3): lc = lc[0]

    grid = gridpp.Grid(lats, lons, topo, lc)
    return grid, lons, lats, vals, leadtime, analysistime, forecasttime, lc, topo


def create_features_data(args):
    '''Create features array which has dimensions for times, stations, variables, and 4 grids.
    In addition create metadata that contains (valid)time and leadtime.''' 
    #Station list and column nearest tells 4 closest grid points for each station
    all_stations = pd.read_csv(args.station_list)
    nearest_column = all_stations.nearest
    nearest_array = np.array(nearest_column.str[1:-1].str.split(',',expand=True).astype(int))

    #Retrieve data in loop to save memory, fetch only one field to python at once
    _, _, data, leadtime, _, forecasttime = read_grib(args.fg_data, False)
    time = [ x.strftime('%Y-%m-%d %H:%M:%S') for x in forecasttime]
    metadata = pd.DataFrame(data = {'leadtime': leadtime,'time': time})
    #Take four closest grid points and save them to features array
    point_values = data.reshape(len(data), -1)[:,nearest_array]
    features = np.empty((len(data), len(all_stations), 20, 4))
    features[:,:,0,:] = point_values
    del data, leadtime, forecasttime
    i = 1
    for param_args in [args.lcc_data,args.mld_data,args.p_data,args.t2_data,args.t850_data,args.tke925_data,args.u10_data,
                       args.u850_data,args.u65_data,args.v10_data,args.v850_data,args.v65_data,args.ugust_data,
                       args.vgust_data,args.z500_data,args.z1000_data,args.z0_data,args.r2_data,args.t0_data]:
        _, _, data, _, _, _ = read_grib(param_args, False)
        #Take four closest grid points
        point_values = data.reshape(len(data), -1)[:,nearest_array]
        features[:,:,i,:] = point_values
        del data
        i += 1

    return features, metadata


def modify_features_for_xgb_model(args, features, metadata):
    '''Select features for given parameter, add station and time features, and add time lagged features'''
    all_stations = pd.read_csv(args.station_list)

    #Interpolate features to station points using 4 closest grid values
    weights  = np.array(all_stations['weights'].str[1:-1].str.split(',',expand=True).astype(float))
    features = np.sum(np.multiply(features, weights[np.newaxis,:,np.newaxis,:]), axis=3)
    
    #Select features for given parameter
    if (args.parameter == "windspeed"): features = features[:,:,0:17]
    if (args.parameter == "windgust"): features = features[:,:,0:17]
    if (args.parameter == "temperature"): features = features[:,:,[1,2,3,4,10,11,17,18,19]]

    #Time lagged features, now 2 lags
    n_lags = 2
    lt_ehto = (metadata['leadtime'] >= n_lags) & (metadata['leadtime'] <= 66)
    features_all_t = features[lt_ehto]
    for i in range(1,n_lags+1):
        features_t = features[(metadata['leadtime'] >= (n_lags-i)) & (metadata['leadtime'] <= (66-i))]
        #Features must be ordered primarily based on leadtime and secondary on time
        features_all_t = np.concatenate((features_all_t, features_t), axis=2)

    #Create time features
    datetime_object = pd.to_datetime(metadata['time'], format = '%Y-%m-%d %H:%M:%S')
    features_time = pd.DataFrame({
        'month_sin': np.sin(datetime_object.dt.month*2*math.pi/12),
        'month_cos': np.cos(datetime_object.dt.month*2*math.pi/12),
        'hour_sin': np.sin(datetime_object.dt.hour*2*math.pi/23),
        'hour_cos': np.cos(datetime_object.dt.hour*2*math.pi/23),
        'leadtime': np.array(metadata['leadtime'])})
    time_features = np.repeat(np.array(features_time)[lt_ehto],len(all_stations),axis=0)

    #Create station features
    features_station = pd.DataFrame({
        'lat': all_stations['LAT'],
        'lon': all_stations['LON'],
        'elev': all_stations['ELEV'],
        'lsm': all_stations['lsm'],
        'z0': all_stations['Z0'],
        'elev_D': all_stations['ELEV']-all_stations['Z0']/9.80655})
    station_features = np.repeat(np.array(features_station)[None,:,:],len(metadata)-n_lags, axis=0).reshape(-1,6)

    #Combine all features to one two dimensional array
    features2 = features_all_t.reshape(-1,features_all_t.shape[2])
    all_features = np.concatenate((features2, time_features, station_features),axis=1)

    return all_features


def ml_predict(all_features, args):
    '''Make xgb prediction and return predicted corrections in list where 
    each element is for one lead time'''
    #Load forecast field
    if (args.parameter == "windspeed"):
        forecast_point = np.sqrt(np.power(all_features[:,7],2) + np.power(all_features[:,10],2))
    elif (args.parameter == "windgust"):
        forecast_point = all_features[:,0]
    elif (args.parameter == "temperature"):
        forecast_point = all_features[:,4] ##HUOM vaihda tarvittaessa

    #Load model
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(args.model)
    quantiles = np.load(args.quantiles)

    #Predict
    xgb_predict = xgb_model.predict(all_features)
    xgb_forecast = forecast_point - xgb_predict

    #Quantile mapping for forecast
    if ((args.parameter == "windspeed") | (args.parameter == "windgust")):
        xgb_forecast[xgb_forecast < 0] = 0
        xgb_forecast_qm = xgb_forecast.copy()
        xgb_forecast_qm[xgb_forecast_qm > 10] = qm.interp_extrap(x=xgb_forecast[xgb_forecast > 10], xp=quantiles['q_ctr'], yp=quantiles['q_obs'])
    elif (args.parameter == "temperature"):
        xgb_forecast_qm = qm.interp_extrap(x=xgb_forecast, xp=quantiles['q_ctr'], yp=quantiles['q_obs'])

    #Predictions back to forecast corrections
    ml_correction = forecast_point - xgb_forecast_qm
    
    # Store data to list where each leadtime is it's own list (leadtimes: +2h..+66h)
    ml_results = []
    leadtimes = all_features[:,-7]

    for j in range(2, 67):
        ml_correction_lt = ml_correction[leadtimes == j]
        ml_results.append(ml_correction_lt)

    return ml_results


def get_points(args):
    '''Create point variable for gridpp interpolation'''
    all_stations = pd.read_csv(args.station_list)
    
    points = gridpp.Points(
        all_stations['LAT'].to_numpy(),
        all_stations['LON'].to_numpy(),
        all_stations['ELEV'].to_numpy(),
        all_stations['lsm'].to_numpy(),
    )

    return points


def interpolate_single_time(
            grid,
            background,
            points,
            obs,
            obs_to_background_variance_ratio,
            pobs,
            structure,
            max_points,
            idx,
            q,
):
    # perform optimal interpolation
    tmp_output = gridpp.optimal_interpolation(
        grid,
        background,
        points,
        obs[idx],
        obs_to_background_variance_ratio,
        pobs,
        structure,
        max_points,
    )

    print(
        "step {} min grid: {:.1f} max grid: {:.1f}".format(
            idx, np.amin(tmp_output), np.amax(tmp_output)
        )
    )

    if q is not None:
        # return index and output, so that the results can
        # later be sorted correctly
        q.put((idx, tmp_output))
    else:
        return tmp_output


def interpolate(grid, points, background, obs, args, lc):
    """Perform optimal interpolation"""
    
    output = []
    
    # create a mask to restrict the modifications only to land area (where lc = 1)
    lc0 = np.logical_not(lc).astype(int)
    
    # Interpolate background data to observation points
    # When bias is gridded then background is zero so pobs is just array of zeros
    pobs = gridpp.nearest(grid, points, background)
    
    # Barnes structure function with horizontal decorrelation length 30km,
    # vertical decorrelation length 200m
    structure = gridpp.BarnesStructure(30000, 200, 0.5)
    
    # Include at most this many "observation points" when interpolating to a grid point
    max_points = 20
    
    # error variance ratio between observations and background
    # smaller values -> more trust to observations
    obs_to_background_variance_ratio = np.full(points.size(), 0.1)
    
    if args.disable_multiprocessing:
        output = [
            interpolate_single_time(
                grid,
                background,
                points,
                obs,
                obs_to_background_variance_ratio,
                pobs,
                structure,
                max_points,
                x,
                None,
            )
            for x in range(len(obs))
        ]
        
    else:
        q = Queue()
        processes = []
        outputd = {}
        
        for i in range(len(obs)):
            processes.append(
                Process(
                    target=interpolate_single_time,
                    args=(
                        grid,
                        background,
                        points,
                        obs,
                        obs_to_background_variance_ratio,
                        pobs,
                        structure,
                        max_points,
                        i,
                        q,
                    ),
                )
            )
            processes[-1].start()
            
        for p in processes:
            # get return values from queue
            # they might be in any order (non-consecutive)
            ret = q.get()
            outputd[ret[0]] = ret[1]
            
        for p in processes:
            p.join()
            
        for i in range(len(obs)):
            # sort return values from 0 to 8
            output.append(outputd[i])
            
    return output


def ml_corrected_forecasts(args, forecasttime, background, diff):
    '''calculate the final ml corrected forecast fields: MEPS - ml_correction
    and make rough qc to forecasts'''
    # Remove leadtimes 0 and 1, because due to lagged features, correction is not made to those
    n_lags = len(forecasttime) - len(diff)
    output = []
    for j in range(0, len(diff)):
        tmp_output = background[j + n_lags] - diff[j]
        # Implement simple QC thresholds
        if args.parameter == "windspeed":
            tmp_output = np.clip(tmp_output, 0, 38)  # max ws same as in oper qc: 38m/s
        elif args.parameter == "windgust":
            tmp_output = np.clip(tmp_output, 0, 50)
        elif args.parameter == "temperature":
            tmp_output = np.clip(tmp_output, 218, 318)
        output.append(tmp_output)

    forecasttime = forecasttime[n_lags:]
    assert len(forecasttime) == len(output)
    
    return output, forecasttime


def write_grib_message(fp, args, analysistime, forecasttime, data):
    pdtn = 70
    tosp = None
    if args.parameter == "temperature":
        levelvalue = 2
        pnum = 0
        pcat = 0
    elif args.parameter == "windspeed":
        pcat = 2
        pnum = 1
        levelvalue = 10
    elif args.parameter == "windgust":
        levelvalue = 10
        pnum = 22
        pcat = 2
        pdtn = 72
        tosp = 2
    # Store different time steps as grib msgs
    for j in range(0, len(data)):
        tdata = data[j]
        forecastTime = int((forecasttime[j] - analysistime).total_seconds() / 3600)
        
        # - For non-aggregated parameters, grib2 key 'forecastTime' is the time of the forecast
        # - For aggregated parameters, it is the start time of the aggregation period. The end of
        #   the period is defined by 'lengthOfTimeRange'
        #   Because snwc is in hourly time steps, reduce forecast time by one
        
        if tosp == 2:
            forecastTime -= 1
            
        assert (tosp is None and j + 2 == forecastTime) or (
            tosp == 2 and j + 1 == forecastTime
        )
        h = ecc.codes_grib_new_from_samples("regular_ll_sfc_grib2")
        ecc.codes_set(h, "tablesVersion", 28)
        ecc.codes_set(h, "gridType", "lambert")
        ecc.codes_set(h, "shapeOfTheEarth", 6)
        ecc.codes_set(h, "Nx", tdata.shape[1])
        ecc.codes_set(h, "Ny", tdata.shape[0])
        ecc.codes_set(h, "DxInMetres", 2370000 / (tdata.shape[1] - 1))
        ecc.codes_set(h, "DyInMetres", 2670000 / (tdata.shape[0] - 1))
        ecc.codes_set(h, "jScansPositively", 1)
        ecc.codes_set(h, "latitudeOfFirstGridPointInDegrees", 50.319616)
        ecc.codes_set(h, "longitudeOfFirstGridPointInDegrees", 0.27828)
        ecc.codes_set(h, "Latin1InDegrees", 63.3)
        ecc.codes_set(h, "Latin2InDegrees", 63.3)
        ecc.codes_set(h, "LoVInDegrees", 15)
        ecc.codes_set(h, "LaDInDegrees", 63.3)
        ecc.codes_set(h, "latitudeOfSouthernPoleInDegrees", -90)
        ecc.codes_set(h, "longitudeOfSouthernPoleInDegrees", 0)
        ecc.codes_set(h, "dataDate", int(analysistime.strftime("%Y%m%d")))
        ecc.codes_set(h, "dataTime", int(analysistime.strftime("%H%M")))
        ecc.codes_set(h, "forecastTime", forecastTime)
        ecc.codes_set(h, "centre", 86)
        ecc.codes_set(h, "generatingProcessIdentifier", args.producer_id) #215 (preop), 214 (prod) 
        ecc.codes_set(h, "discipline", 0)
        ecc.codes_set(h, "parameterCategory", pcat)
        ecc.codes_set(h, "parameterNumber", pnum)
        ecc.codes_set(h, "productDefinitionTemplateNumber", pdtn)
        if tosp is not None:
            ecc.codes_set(h, "typeOfStatisticalProcessing", tosp)
            ecc.codes_set(h, "lengthOfTimeRange", 1)
            ecc.codes_set(
                h, "yearOfEndOfOverallTimeInterval", int(forecasttime[j].strftime("%Y"))
            )
            ecc.codes_set(
                h,
                "monthOfEndOfOverallTimeInterval",
                int(forecasttime[j].strftime("%m")),
            )
            ecc.codes_set(
                h, "dayOfEndOfOverallTimeInterval", int(forecasttime[j].strftime("%d"))
            )
            ecc.codes_set(
                h, "hourOfEndOfOverallTimeInterval", int(forecasttime[j].strftime("%H"))
            )
            ecc.codes_set(h, "minuteOfEndOfOverallTimeInterval", 0)
            ecc.codes_set(h, "secondOfEndOfOverallTimeInterval", 0)
        ecc.codes_set(h, "typeOfFirstFixedSurface", 103)
        ecc.codes_set(h, "scaledValueOfFirstFixedSurface", levelvalue)
        ecc.codes_set(h, "packingType", "grid_ccsds")
        ecc.codes_set(h, "indicatorOfUnitOfTimeRange", 1)  # hours
        ecc.codes_set(h, "typeOfGeneratingProcess", 2)  # deterministic forecast
        ecc.codes_set(h, "typeOfProcessedData", 2)  # analysis and forecast products
        ecc.codes_set_values(h, tdata.flatten())
        ecc.codes_write(h, fp)
    ecc.codes_release(h)
            

def write_grib(args, analysistime, forecasttime, data):
    if args.output.startswith("s3://"):
        openfile = fsspec.open(
            "simplecache::{}".format(args.output),
            "wb",
            s3={
                "anon": False,
                "key": os.environ["S3_ACCESS_KEY_ID"],
                "secret": os.environ["S3_SECRET_ACCESS_KEY"],
                "client_kwargs": {"endpoint_url": "https://lake.fmi.fi"},
            },
        )
        with openfile as fpout:
            write_grib_message(fpout, args, analysistime, forecasttime, data)
    else:
        with open(args.output, "wb") as fpout:
            write_grib_message(fpout, args, analysistime, forecasttime, data)
            
    print(f"Wrote file {args.output}")



