# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:13:36 2023

@author: ylinenk
"""

import numpy as np
import copy
import time
import argparse
from helper_functions import create_features_data
from helper_functions import modify_features_for_xgb_model
from helper_functions import ml_predict
from helper_functions import read_grid
from helper_functions import get_points
from helper_functions import interpolate
from helper_functions import write_grib
from helper_functions import ml_corrected_forecasts

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter", action="store", type=str, required=True)
    parser.add_argument("--topography_data", action="store", type=str, required=True)
    parser.add_argument("--landseacover_data", action="store", type=str, required=True)
    parser.add_argument("--fg_data", action="store", type=str, required=True)
    parser.add_argument("--lcc_data", action="store", type=str, required=True)
    parser.add_argument("--mld_data", action="store", type=str, required=True)
    parser.add_argument("--p_data", action="store", type=str, required=True)
    parser.add_argument("--t2_data", action="store", type=str, required=True)
    parser.add_argument("--t850_data", action="store", type=str, required=True)
    parser.add_argument("--tke925_data", action="store", type=str, required=True)
    parser.add_argument("--u10_data", action="store", type=str, required=True)
    parser.add_argument("--u850_data", action="store", type=str, required=True)
    parser.add_argument("--u65_data", action="store", type=str, required=True)
    parser.add_argument("--v10_data", action="store", type=str, required=True)
    parser.add_argument("--v850_data", action="store", type=str, required=True)
    parser.add_argument("--v65_data", action="store", type=str, required=True)
    parser.add_argument("--ugust_data", action="store", type=str, required=True)
    parser.add_argument("--vgust_data", action="store", type=str, required=True)
    parser.add_argument("--z500_data", action="store", type=str, required=True)
    parser.add_argument("--z1000_data", action="store", type=str, required=True)
    parser.add_argument("--z0_data", action="store", type=str, required=True)
    parser.add_argument("--r2_data", action="store", type=str, required=True)
    parser.add_argument("--t0_data", action="store", type=str, required=True)
    parser.add_argument("--model", action="store", type=str, required=True)
    parser.add_argument("--quantiles", action="store", type=str, required=True)
    parser.add_argument("--station_list", action="store", type=str, required=True)
    parser.add_argument("--output", action="store", type=str, required=True)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--disable_multiprocessing", action="store_true", default=False)
    
    args = parser.parse_args()
    
    allowed_params = ["temperature", "windspeed", "windgust"]
    if args.parameter not in allowed_params:
        print("Error: parameter must be one of: {}".format(allowed_params))
        sys.exit(1)
        
    return args

                                                                                                    
def main():
    args = parse_command_line()

    #Read NWP data and create fetures array
    st = time.time()
    print("lol")
    print(st)
    features, metadata = create_features_data(args)
    print("Reading NWP data for", args.parameter, "takes:", round(time.time()-st, 1), "seconds")

    #ML prediction
    mlt = time.time()
    all_features = modify_features_for_xgb_model(args, features, metadata)
    ml_predictions = ml_predict(all_features, args)
    print("Producing ML forecasts takes:", round(time.time()-mlt, 1), "seconds")

    #Gridding
    oit = time.time()
    grid, lons, lats, background, leadtimes, analysistime, forecasttime, lc, topo = read_grid(args)
    background0 = copy.copy(background)
    background0[background0 != 0] = 0
    points = get_points(args)
    diff = interpolate(grid, points, background0[0], ml_predictions, args, lc)
    print("Interpolating forecasts takes:", round(time.time()-oit, 1), "seconds")

    #Write corrected forecasts to grib file
    output, forecasttime = ml_corrected_forecasts(args, forecasttime, background, diff)
    write_grib(args, analysistime, forecasttime, output)

    if args.plot:
        from plotting_functions import plot_results
        print("Plotting results:")
        plt = time.time()
        plot_results(args, lons, lats, background, diff, output, analysistime, forecasttime, leadtimes)
        print("Plotting results takes:", round(time.time()-plt, 1), "seconds")

if __name__ == "__main__":
    main()
