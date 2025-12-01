import os
from osgeo import gdal # install with `pip install gdal==3.8.4`
import numpy as np
from importlib.resources import files
from pathlib import Path

# Note from Cameron, 2025-10-24:
# Problems with gdal? Read https://pypi.org/project/GDAL/
# If you get an error saying something along the lines of 'ImportError: cannot import name '_gdal_array' from 'osgeo' (/usr/local/lib/python3.12/dist-packages/osgeo/__init__.py)' 
# then try the command 'pip install --no-cache --force-reinstall gdal[numpy]=="$(gdal-config --version).*"' to reinstall it.


def MeanDEM(pointUL, pointDR, dem_path: Path = None) -> float:
    """
    Calculate the average elevation of the area where the image is located.

    :param pointUL: Upper left corner of the lat/lon array
    :param pointDR: Lower right corner of the lat/lon array

    :return: Mean elevation of the area where the image was captured
    """


    if dem_path is None:

        script_dir = os.path.dirname(os.path.abspath(__file__))
        dem_path = os.path.join(script_dir, "GMTED2km.tif")

    else:
        dem_path = Path(dem_path).absolute()

    try:
        DEMIDataSet = gdal.Open(str(dem_path))
    except Exception as e:
        raise e

    DEMBand = DEMIDataSet.GetRasterBand(1)
    geotransform = DEMIDataSet.GetGeoTransform()

    # DEM Resolution
    pixelWidth = geotransform[1]
    pixelHight = geotransform[5]

    # DEM start point: top left corner, X: longitude, Y: latitude
    originX = geotransform[0]
    originY = geotransform[3]

    # Location of the upper left corner of the study area in the DEM matrix
    yoffset1 = int((originY - pointUL['lat']) / pixelWidth)
    xoffset1 = int((pointUL['lon'] - originX) / (-pixelHight))

    # Location of the lower right corner of the study area in the DEM matrix
    yoffset2 = int((originY - pointDR['lat']) / pixelWidth)
    xoffset2 = int((pointDR['lon'] - originX) / (-pixelHight))

    # Number of ranks of the matrix in the study area
    xx = xoffset2 - xoffset1
    yy = yoffset2 - yoffset1


    # Handle single point look up
    if xx == 0:
        xx = xx + 1
    
    if yy == 0:
        yy = yy + 1

    #DEMBand.ReadAsArray(x, y, 1, 1)[0, 0]

    # Read data from the study area and calculate elevations
    DEMRasterData = DEMBand.ReadAsArray(xoffset1, yoffset1, xx, yy)
    #DEMRasterData = DEMBand.ReadAsArray(xoffset1, yoffset1, 1, 1)

    MeanAltitude = np.mean(DEMRasterData)

    return MeanAltitude



if __name__ == "__main__":

    #pointUL = {'lat': np.float32(47.712517), 'lon': np.float32(10.7741785)}
    #pointDR = {'lat': np.float32(42.814625), 'lon': np.float32(14.932583)}
    
    pointUL = {'lat': np.float32(47.239), 'lon': np.float32(14.932)}
    pointDR = {'lat': np.float32(47.239), 'lon': np.float32(14.932)}



    MeanAltitude = MeanDEM(pointUL, pointDR)

    print(MeanAltitude)
