# Modified from pip to have as local directory
from hypso.classification.WaterDetection.waterdetect.Common import DWConfig
from hypso.classification.WaterDetection.waterdetect.Image import DWImageClustering
from osgeo import gdal, osr
import numpy as np
from importlib.resources import files
from typing import Literal


def ndwi_watermask(sat_obj, product_to_use: Literal["L1C", "L1B", "L2-ACOLITE", "L2-6SV1"] = "L1C") -> None:
    """
    Compute the Water Mask using the Normalized Difference Water Index (NDWI). This method was originally developed by
    Mauricio Cordeiro (https://github.com/cordmaur/WaterDetect)

    :param sat_obj: Hypso satellite object
    :param product_to_use: String of product to use for calculating the water mask. Default and recommended: "L1C"

    :return: No Return
    """
    sat_obj.find_geotiffs()

    # TODO: Confirm why L1C gives better results than L2
    # Check if full (120 band) tiff exists
    if sat_obj.l1cgeotiffFilePath is None and (product_to_use == "L1C" or product_to_use == "L1B"):
        raise Exception("No Full-Band L1C GeoTiff")

    if sat_obj.l2geotiffFilePath is None and product_to_use == "L2":
        raise Exception("No Full-Band L2 GeoTiff")

    cube_selected = None
    if "L2" in product_to_use:
        ds = gdal.Open(str(sat_obj.l2geotiffFilePath))
        data_mask = ds.ReadAsArray()
        data = np.ma.masked_where(data_mask == 0, data_mask)
        cube_selected = np.rot90(data.transpose((1, 2, 0)), k=2)

    elif product_to_use == "L1C" or product_to_use == "L1B":
        ds = gdal.Open(str(sat_obj.l1cgeotiffFilePath))
        data_mask = ds.ReadAsArray()
        data = np.ma.masked_where(data_mask == 0, data_mask)
        cube_selected = np.rot90(data.transpose((1, 2, 0)), k=2)

        cube_selected = sat_obj.l1b_cube

        # TODO: Check which one is better geotiff or from netcdf
        # We can read the geotiff or just use l1b from the netcdf without masking
        # cube_selected = sat_obj.l1b_cube

    print("\n\n-------  Naive-Bayes Water Mask Detector  ----------")

    water_config_file = files(
        'hypso.classification').joinpath('WaterDetection/WaterDetect.ini')

    config = DWConfig(config_file=water_config_file)
    print(config.clustering_bands)
    print(config.detect_water_cluster)

    # Band 3 in Sentinel-2 is centered at 560nm with a FWHM: 34.798nm
    # Band 560.6854258053552 at index 49 is the closes Hypso equivalent
    b3 = cube_selected[:, :, 49]

    # Band 8 in Sentinel-2 is NIR centered at 835nm with a FWHM: 104.784n8
    # Hypso last 2 bands (maybe noisy) are at 799.10546171nm & 802.51814719nm at index 118 and 119
    nir = cube_selected[:, :, 118]

    # Division by 10000 may not be needed
    bands = {'Green': b3, 'Nir': nir}
    wmask = DWImageClustering(bands=bands, bands_keys=[
        'Nir', 'ndwi'], invalid_mask=None, config=config)

    mask = wmask.run_detect_water()

    mask = wmask.water_mask

    # Mask Adjustment
    boolMask = mask.copy()
    boolMask[boolMask != 1] = 0
    boolMask = boolMask.astype(bool)

    sat_obj.waterMask = boolMask


def threshold_watermask(sat_obj, threshold_val: float = 2.9) -> None:
    """
    Simple Threshold WaterMask

    :param sat_obj: Hypso satellite object
    :param threshold_val: Threshold value for pixel contrast. Default is 2.9
    :return:
    """
    # # TODO: Improve Water Map Selection (Non threshold dependant)
    def get_is_water_map(cube, wl, binary_threshold=threshold_val):
        C1 = np.argmin(abs(wl - 460))
        C2 = np.argmin(abs(wl - 650))
        ret = np.zeros([cube.shape[0], cube.shape[1]])
        for xx in range(cube.shape[0]):
            for yy in range(cube.shape[1]):
                spectra = cube[xx, yy, :]
                ret[xx, yy] = spectra[C1] / spectra[C2]

        return ret > binary_threshold

    boolMask = get_is_water_map(sat_obj.l1b_cube, sat_obj.wavelengths)

    sat_obj.waterMask = boolMask
