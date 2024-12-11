import pyproj
import pandas as pd
import numpy as np
import argparse

"""
Grid metadata for MEPS domain
"""
nx = 949
ny = 1069
dx = 2500
dy = 2500
firstPoint = (50.319616,0.27828)

lcc_proj = "+proj=lcc +lat_0=63.3 +lon_0=15 +lat_1=63.3 +lat_2=63.3 +a=6371229 +b=6371229 +units=m +no_defs"

latlon_to_lcc = pyproj.Transformer.from_crs("EPSG:4326",lcc_proj)
x_0,y_0 = latlon_to_lcc.transform(firstPoint[0],firstPoint[1])

def main():
    """
    Create the argument parser
    """
    parser = argparse.ArgumentParser(description="Find surrounding grid points for a station, calculate interpolation weights for station point and add those as new columns to the output file.")
    
    """
    Add arguments
    """
    parser.add_argument('-i', '--infile', type=str, help="Filepath to input station list", required=True)
    parser.add_argument('-o', '--outfile', type=str, help="Filepath to output station list", required=True)
    
    """
    Parse the arguments
    """
    args = parser.parse_args()

    infile = args.infile
    outfile = args.outfile

    """
    load station list
    """
    sl = pd.read_csv(infile)

    """
    Assert station list contains LAT and LON
    """
    assert 'LAT' in sl.columns and 'LON' in sl.columns

    """
    Find four nearest grid points for each station and 
    calculate interpolation weights for bilinear 
    interpolation to station coordinates
    """
    coords = []
    indexes = []
    weights = []
    for i in range(len(sl)):
        x,y = latlon_to_lcc.transform(sl['LAT'][i],sl['LON'][i])
        coords.append([(x-x_0)/dx,(y-y_0)/dy])
        dxi,xi = np.modf((x-x_0)/dx)
        dyj,yj = np.modf((y-y_0)/dy)
        i_0 = int(xi+949*yj)
        indexes.append([i_0,i_0+1,i_0+949,i_0+949+1])
        weights.append([(1-dxi)*(1-dyj),dxi*(1-dyj),(1-dxi)*dyj,dxi*dyj])

    sl['nearest'] = indexes
    sl['weights'] = weights

    sl.to_csv(outfile,index=False)

if __name__ == "__main__":
    main()
