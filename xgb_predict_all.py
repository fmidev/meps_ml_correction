# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:13:36 2023

@author: ylinenk
"""

import sys
import copy
import time
import argparse
from helper_functions import create_features_data
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
    parser.add_argument("--td2_data", action="store", type=str, required=True)
    parser.add_argument("--tmax_data", action="store", type=str, required=True)
    parser.add_argument("--tmin_data", action="store", type=str, required=True)
    parser.add_argument("--model_ws", action="store", type=str, required=True)
    parser.add_argument("--quantiles_ws", action="store", type=str, required=True)
    parser.add_argument("--model_wg", action="store", type=str, required=True)
    parser.add_argument("--quantiles_wg", action="store", type=str, required=True)
    parser.add_argument("--model_ta", action="store", type=str, required=True)
    parser.add_argument("--quantiles_ta", action="store", type=str, required=True)
    parser.add_argument("--model_td", action="store", type=str, required=True)
    parser.add_argument("--quantiles_td", action="store", type=str, required=True)
    parser.add_argument("--model_tmax", action="store", type=str, required=True)
    parser.add_argument("--quantiles_tmax", action="store", type=str, required=True)
    parser.add_argument("--model_tmin", action="store", type=str, required=True)
    parser.add_argument("--quantiles_tmin", action="store", type=str, required=True)
    parser.add_argument("--stations_list_ws", action="store", type=str, required=True)
    parser.add_argument("--stations_list_wg", action="store", type=str, required=True)
    parser.add_argument("--stations_list_ta", action="store", type=str, required=True)
    parser.add_argument("--stations_list_td", action="store", type=str, required=True)
    parser.add_argument("--analysis_time", action="store", type=str, required=True)
    parser.add_argument("--producer_id", action="store", type=int, required=True)
    parser.add_argument("--output", action="store", type=str, required=True)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--disable_multiprocessing", action="store_true", default=False)
    
    args = parser.parse_args()
    
    allowed_params = ["dewpoint","temperature", "windspeed", "windgust"]
    if args.parameter not in allowed_params:
        print("Error: parameter must be one of: {}".format(allowed_params))
        sys.exit(1)
        
    return args

                                                                                                    
def main():
    args = parse_command_line()

    #Read NWP data and create fetures array
    st = time.time()
    features, metadata = create_features_data(args, args.parameter)   
    if (args.parameter == "windgust"): #For windgust, we need also features from wind speed
        features_ws, _ = create_features_data(args, "windspeed")
    elif (args.parameter == "dewpoint"): #For dewpoint, we need also features from temperature
        features_ta, _ = create_features_data(args, "temperature")
    print("Reading NWP data for", args.parameter, "takes:", round(time.time()-st, 1), "seconds")

    #ML prediction
    mlt = time.time()
    ml_predictions = ml_predict(args, features, metadata, args.parameter)
    if (args.parameter == "windgust"):
        ml_predictions_ws = ml_predict(args, features_ws, metadata, "windspeed")
    elif (args.parameter == "dewpoint"):
        ml_predictions_ta = ml_predict(args, features_ta, metadata, "temperature")
    print("Producing ML forecasts takes:", round(time.time()-mlt, 1), "seconds")

    #Gridding
    oit = time.time()
    grid, lons, lats, background, leadtimes, analysistime, forecasttime, lc, topo = read_grid(args, args.parameter)
    background0 = copy.copy(background)
    background0[background0 != 0] = 0
    points = get_points(args, args.parameter)
    diff = interpolate(grid, points, background0[0], ml_predictions, args, lc)
    output, forecasttime = ml_corrected_forecasts(forecasttime, background, diff, args.parameter)
    #Make value check for wind gust and dewpoint
    if (args.parameter == "windgust"):
        _, _, _, background_ws, _, _, forecasttime, _, _ = read_grid(args, "windspeed")
        points_ws = get_points(args, "windspeed")
        diff_ws = interpolate(grid, points_ws, background0[0], ml_predictions_ws, args, lc)
        output_ws, forecasttime = ml_corrected_forecasts(forecasttime, background_ws, diff_ws, "windspeed")
        #Set that output of wind gust cant be lower than output of wind speed
        for i in range(0, len(output)):
            wg_lt_ws = output[i] < output_ws[i]
            output[i][wg_lt_ws] = output_ws[i][wg_lt_ws] + 0.0001
    elif (args.parameter == "dewpoint"):
        _, _, _, background_ta, _, _, forecasttime, _, _ = read_grid(args, "temperature")
        points_ta = get_points(args, "temperature")
        diff_ta = interpolate(grid, points_ta, background0[0], ml_predictions_ta, args, lc)
        output_ta, forecasttime = ml_corrected_forecasts(forecasttime, background_ta, diff_ta, "temperature")
        #Set that output of dewpoint cant be higher than output of temperature
        for i in range(0, len(output)):
            td_gt_ta = output[i] > output_ta[i]
            output[i][td_gt_ta] = output_ta[i][td_gt_ta] - 0.0001
    print("Interpolating forecasts takes:", round(time.time()-oit, 1), "seconds")

    #Write corrected forecasts to grib file
    write_grib(args, analysistime, forecasttime, output)

    if args.plot:
        from plotting_functions import plot_results
        print("Plotting results:")
        plt = time.time()
        plot_results(args, lons, lats, background, diff, output, analysistime, forecasttime, leadtimes)
        print("Plotting results takes:", round(time.time()-plt, 1), "seconds")

if __name__ == "__main__":
    main()
