import os
from osgeo import gdal
import numpy as np
from importlib.resources import files

def MeanDEM(pointUL, pointDR):
    '''
    Calculate the average elevation of the area where the image is located.
    '''

    dem_path = files(
        'hypso.atmospheric').joinpath(f'GMTED2km.tif')

    try:
        DEMIDataSet = gdal.Open(dem_path)
    except Exception as e:
        pass

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
    # Read data from the study area and calculate elevations
    DEMRasterData = DEMBand.ReadAsArray(xoffset1, yoffset1, xx, yy)

    MeanAltitude = np.mean(DEMRasterData)
    return MeanAltitude
