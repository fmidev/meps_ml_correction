# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:53:33 2023

@author: ylinenk
"""

import os
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import cartopy
from datetime import datetime as dt
import numpy as np
import matplotlib
    
def plot_results(args, lons, lats, background, diff, output, analysistime, forecasttime, leadtimes):
    '''plot raw and ml corrected meps forecasts on map, and their difference (correction)'''
    outfile = f"figures/{dt.strftime(analysistime, '%Y%m%d%H')}/"
    os.makedirs(outfile, exist_ok=True)
    
    if (args.parameter == "windspeed"):
        variable_min = 0
        variable_max = 50
    elif (args.parameter == "windgust"):
        variable_min = 0
        variable_max = 50
    else:
        variable_min = min(np.min(output),np.min(background))
        variable_max = max(np.max(output),np.max(background))

    N=20
    base_cmaps = ['tab20b','tab20c']
    colors = np.concatenate([plt.get_cmap(name)(np.linspace(0,1,N)) for name in base_cmaps])
    cmap40 = matplotlib.colors.ListedColormap(colors)
        
    for i in range(0,len(diff)):
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 6), dpi=80)
        m = Basemap(width=2000000, height=2300000,resolution='l', rsphere=(6378137.00,6356752.3142),projection='lcc', ellps='WGS84',lat_1=64.8, lat_2=64.8, lat_0=62.0, lon_0=16.0, ax=axs[0])
        m.drawcountries(linewidth=1.0)
        m.drawcoastlines(1.0)
        d = background[i+2]
        x, y = m(lons, lats)
        cm = m.pcolormesh(x, y, d, cmap=cmap40, vmin=variable_min, vmax=variable_max)
        axs[0].set_title("MEPS forecast")
        plt.colorbar(cm, fraction=0.033, pad=0.04, orientation="horizontal")

        m = Basemap(width=2000000, height=2300000,resolution='l', rsphere=(6378137.00,6356752.3142),projection='lcc', ellps='WGS84',lat_1=64.8, lat_2=64.8, lat_0=62.0, lon_0=16.0, ax=axs[1])
        m.drawcountries(linewidth=1.0)
        m.drawcoastlines(1.0)
        d = diff[i]
        x, y = m(lons, lats)
        vmax = max(abs(np.min(diff[i])),np.max(diff[i]))
        cm = m.pcolormesh(x, y, d, cmap="seismic_r", vmin=-vmax, vmax=vmax)
        axs[1].set_title("XGB correction")
        axs[1].annotate('min: ' + str(round(np.min(diff[i]),1)), xy=(0.03, 0.96), xycoords='axes fraction')
        axs[1].annotate('max: ' + str(round(np.max(diff[i]),1)), xy=(0.03, 0.92), xycoords='axes fraction')
        plt.colorbar(cm, fraction=0.033, pad=0.04, orientation="horizontal")

        m = Basemap(width=2000000, height=2300000,resolution='l', rsphere=(6378137.00,6356752.3142),projection='lcc', ellps='WGS84',lat_1=64.8, lat_2=64.8, lat_0=62.0, lon_0=16.0, ax=axs[2])
        m.drawcountries(linewidth=1.0)
        m.drawcoastlines(1.0)
        d = output[i]
        x, y = m(lons, lats)
        cm = m.pcolormesh(x, y, d, cmap=cmap40, vmin=variable_min, vmax=variable_max)
        axs[2].set_title("XGB forecast")
        plt.colorbar(cm, fraction=0.033, pad=0.04, orientation="horizontal")
        plt.suptitle(f"{args.parameter} forecast {dt.strftime(forecasttime[i], '%Y-%m-%d %H:%M:%S')}\n{dt.strftime(analysistime, '%Y-%m-%d %H:%M:%S')} (+{str(leadtimes[i+2]).zfill(2)}h)")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.4, hspace=None)
        forecast_outfile = outfile + f"{args.parameter}_{dt.strftime(analysistime, '%Y%m%d%H')}+{str(leadtimes[i+2]).zfill(2)}h.png"
        plt.savefig(forecast_outfile, dpi=150, bbox_inches='tight', pad_inches=0.2)
        plt.close()

