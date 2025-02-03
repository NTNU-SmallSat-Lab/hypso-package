import numpy as np
import pyproj as prj
import matplotlib.path as mplpath
import shapely.geometry as sg
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo import osr
from pathlib import Path
# GIS
import math as m
import threading
import scipy.interpolate as si

from hypso.georeference import georef as gref
from hypso.utils import find_file, HSI2RGB
from typing import Literal, List

DEBUG = False
EXPORT_SINGLE_BANDS = False

r_band_index = 61  # Old 59 (120-59=61)
g_band_index = 50  # Old 70 (120-70=50)
b_band_index = 31  # Old 89 (120-89=31)

'''
def generate_rgb_geotiff(satObj, overwrite: bool = False) -> None:
    """
    Generate RGB GeoTiff image

    :param satObj: Hypso satellite object.
    :param overwrite: If true, overwrite the previously generated RGB GeoTiff image

    :return: No return
    """
    existing_rgb_geotiff = find_file(satObj.info["top_folder_name"], "-rgb", ".tif")

    if existing_rgb_geotiff is not None:
        if overwrite:
            existing_rgb_geotiff.unlink(missing_ok=True)
        else:
            return
    # GeoTiff Output ------------------------------------------------------------------------
    top_folder_name = satObj.info["top_folder_name"]
    capture_name = str(satObj.info["capture_name"])
    geotiff_folder_path = Path(top_folder_name, "geotiff")
    output_path_rgb_tif = Path(geotiff_folder_path, capture_name + '-rgb.tif')
    output_path_rgba_tif = Path(geotiff_folder_path, capture_name + '-rgba_8bit.tif')

    # Select data for RGB GeoTiff --------------------------------------------------------
    cube_data = satObj.l1b_cube

    # Define bands to export and RGB Bands (Need to be the same to interpolate) ---------------
    extra_band_index = (g_band_index - 4) // 4
    bands = [extra_band_index, r_band_index, g_band_index, b_band_index]
    rgb_band_indices = [r_band_index, g_band_index, b_band_index]

    # Get Geometric Information----------------------------------------------------------------
    grid_dims, geotransform, destination_epsg, grid_data_all_bands, contain_mask = generate_geotiff(satObj, bands,
                                                                                                    cube_data)

    # Max Value RGB To Normalize Images -------------------------------------------------------
    max_value_rgb = 0.0
    for rgb_band_index in rgb_band_indices:
        max_current_band = grid_data_all_bands[:, :, rgb_band_index].max()
        if max_current_band > max_value_rgb:
            max_value_rgb = max_current_band

    # Geotiff Objects ------------------------------------------------------------------------------------------
    # some Geotiff reference info:
    # https://gis.stackexchange.com/questions/380607/how-to-geo-reference-a-tif-image-knowing-corner-coordinates
    # https://gis.stackexchange.com/questions/275125/expressing-raster-rotation-with-gdal-geotransform-python

    dst_ds = gdal.GetDriverByName('GTiff').Create(
        str(output_path_rgb_tif), grid_dims[0], grid_dims[1], 3, gdal.GDT_Byte)  # Old: GDT_UInt16

    dst_ds_alpha_channel = gdal.GetDriverByName('GTiff').Create(
        str(output_path_rgba_tif), grid_dims[0], grid_dims[1], 4, gdal.GDT_Byte)

    # RGB ------------------------------------------------------
    dst_ds.SetGeoTransform(geotransform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    srs.ImportFromEPSG(destination_epsg)
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
    for i, rgb_band_index in enumerate(rgb_band_indices):
        dst_ds.GetRasterBand(
            i + 1).WriteArray(grid_data_all_bands[:, :, rgb_band_index])
    dst_ds.FlushCache()  # write to disk

    # RGBA -------------------------------------------------------
    # Alpha mask for transparency in the RGBA geotiff
    alpha_mask = np.zeros([grid_dims[1], grid_dims[0]])
    alpha_mask.shape = [grid_dims[1] * grid_dims[0]]
    alpha_mask[contain_mask] = 255
    alpha_mask.shape = [grid_dims[1], grid_dims[0]]
    alpha_mask = alpha_mask[:, ::-1]

    dst_ds_alpha_channel.SetGeoTransform(geotransform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    srs.ImportFromEPSG(destination_epsg)
    dst_ds_alpha_channel.SetProjection(
        srs.ExportToWkt())  # export coords to file

    for i, rgb_band_index in enumerate(rgb_band_indices):
        # dst_ds_alpha_channel.GetRasterBand(
        #     i + 1).WriteArray(255.0 * grid_data_all_bands[:, :, rgb_band_index] / max_value_rgb)

        dst_ds_alpha_channel.GetRasterBand(
            i + 1).WriteArray(255.0 * grid_data_all_bands[:, :, rgb_band_index] / max_value_rgb)
    dst_ds_alpha_channel.GetRasterBand(4).WriteArray(alpha_mask)
    dst_ds_alpha_channel.FlushCache()  # write to disk

    print('Done RGB/RGBA Geotiff')

'''
    

'''
def generate_full_geotiff(satObj, product: Literal["L1C", "L2-6SV1", "L2-ACOLITE"] = "L1C") -> None:
    """
    Generate Full 120 Band GeoTiff image

    :param satObj: Hypso satellite object.
    :param product: Product to generate it can be either "L1C", "L2-6SV1", "L2-ACOLITE"

    :return: No return
    """

    # GeoTiff Output ------------------------------------------------------------------------
    top_folder_name = satObj.info["top_folder_name"]
    capture_name = str(satObj.info["capture_name"])
    geotiff_folder_path = Path(top_folder_name, "geotiff")

    # L2A is priority, if we dont have it, use L1B ---------------------------------------------------
    cube_data = None
    output_path_full_tif = None
    if "L2" in product:
        try:
            L2_key = product.split("-")[1].upper()
        except IndexError:
            L2_key = None

        # Find L2 product file, if exists, dont run the entire process again
        l2geotiffFilePath = find_file(satObj.info["top_folder_name"], f"-full_L2_{L2_key}", ".tif")
        if l2geotiffFilePath is not None:
            return

        # If we dont have it, create the new path
        cube_data = satObj.l2a_cube[L2_key]
        output_path_full_tif = Path(geotiff_folder_path, capture_name + f'-full_L2_{L2_key}.tif')
    elif product == "L1C":
        # Find L1C product file, if exists, dont run the entire process again
        l1cgeotiffFilePath = find_file(satObj.info["top_folder_name"], "-full_L1C", ".tif")
        if l1cgeotiffFilePath is not None:
            return

        # If we dont have it, create the new path
        cube_data = satObj.l1b_cube
        output_path_full_tif = Path(geotiff_folder_path, capture_name + '-full_L1C.tif')

    bands = [i for i in range(120)]

    # Get Geometric Information----------------------------------------------------------------
    grid_dims, geotransform, destination_epsg, grid_data_all_bands, _ = generate_geotiff(satObj, bands, cube_data)

    # Geotiff Objects ------------------------------------------------------------------------------------------
    # some Geotiff reference info:
    # https://gis.stackexchange.com/questions/380607/how-to-geo-reference-a-tif-image-knowing-corner-coordinates
    # https://gis.stackexchange.com/questions/275125/expressing-raster-rotation-with-gdal-geotransform-python

    dst_ds_full = gdal.GetDriverByName('GTiff').Create(
        str(output_path_full_tif), grid_dims[0], grid_dims[1], 120, gdal.GDT_Float64)  # Used to be 64

    # Full Band Tiff --------------------------------
    dst_ds_full.SetGeoTransform(geotransform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    srs.ImportFromEPSG(destination_epsg)
    dst_ds_full.SetProjection(
        srs.ExportToWkt())  # export coords to file

    for i, band_index in enumerate(bands):
        dst_ds_full.GetRasterBand(
            i + 1).WriteArray(grid_data_all_bands[:, :, band_index])

    dst_ds_full.FlushCache()  # write to disk

    print('Done Full Geotiff')
'''

'''
def generate_geotiff(satObj, bands: list, cube_data: np.ndarray) -> None:
    """
    Generate geotiff (RGB or Full 120 bands)

    :param satObj: Hypso Satellite object
    :param bands: List of band indices
    :param cube_data: Numpy array of the cube array to include in the GeoTiff

    :return: No return
    """
    print('Generating Geotiff ************************************')

    # Setting some default values -------------------------------------------------
    frame_count = satObj.info["frame_count"]
    hypso_height = satObj.info["image_height"]
    hypso_height_sensor = 1216
    hypso_width_sensor = 1936

    hypso_width = 120
    band_count = hypso_width

    pixels_lat = satObj.latitudes
    pixels_lon = satObj.longitudes

    # Setup Paths -----------------------------------------------------------
    top_folder_name = satObj.info["top_folder_name"]
    capture_name = str(satObj.info["capture_name"])
    geotiff_folder_path = Path(top_folder_name, "geotiff")
    geotiff_folder_path.mkdir(parents=True, exist_ok=True)
    output_path_geo_info = Path(top_folder_name, capture_name + '-geometric-meta-info-full.txt')

    # rojecting pixel geodetic to ma --------------------------------------------
    print('  Projecting pixel geodetic to map ...')
    bbox_geodetic = [np.min(pixels_lat), np.max(
        pixels_lat), np.min(pixels_lon), np.max(pixels_lon)]
    print('   ', bbox_geodetic)
    utm_crs_list = prj.database.query_utm_crs_info(datum_name="WGS 84",
                                                   area_of_interest=prj.aoi.AreaOfInterest(
                                                       west_lon_degree=bbox_geodetic[2],
                                                       south_lat_degree=bbox_geodetic[0],
                                                       east_lon_degree=bbox_geodetic[3],
                                                       north_lat_degree=bbox_geodetic[1], )
                                                   )
    print(f'    using UTM map: ' +
          utm_crs_list[0].name, 'EPSG:', utm_crs_list[0].code)

    # crs_25832 = prj.CRS.from_epsg(25832) # UTM32N
    # crs_32717 = prj.CRS.from_epsg(32717) # UTM17S
    crs_4326 = prj.CRS.from_epsg(4326)  # Unprojected [(lat,lon), probably]
    source_crs = crs_4326
    destination_epsg = int(utm_crs_list[0].code)
    destination_crs = prj.CRS.from_epsg(destination_epsg)
    latlon_to_proj = prj.Transformer.from_crs(source_crs, destination_crs)

    pixel_coords_map = np.zeros([frame_count, hypso_height, 2])
    for i in range(frame_count):
        for j in range(hypso_height):
            pixel_coords_map[i, j, :] = latlon_to_proj.transform(
                pixels_lat[i, j], pixels_lon[i, j])

    dg_bounding_path = np.concatenate(
        (pixel_coords_map[:, 0, :], pixel_coords_map[::-1, (hypso_height - 1), :]))

    # Direct Georeferencing outline --------------------------------------------------------
    boundingpath = mplpath.Path(dg_bounding_path)
    boundingpath_area = sg.Polygon(dg_bounding_path).area

    with open(output_path_geo_info, 'a') as f:
        f.write(
            f'Imaged area (square kilometers): {boundingpath_area / 1000000.0:09.5f}\n')
    print(
        f'    Area covered by image: {boundingpath_area} m^2 = {boundingpath_area / 1000000.0} km^2')

    if DEBUG:
        plt.scatter(pixels_lon[:, 0], pixels_lat[:, 0], label='bot')
        plt.scatter(pixels_lon[:, (hypso_height // 2 - 1)], pixels_lat[:, (hypso_height // 2 - 1)], label='mid')
        plt.scatter(pixels_lon[:, (hypso_height - 1)], pixels_lat[:, (hypso_height - 1)], label='top')
        plt.scatter(pixels_lon[0, :], pixels_lat[0, :], label='first frame')
        plt.scatter(pixels_lon[955, :], pixels_lat[955, :], label='last frame')
        plt.grid()
        plt.legend()
        plt.show()
        # time lines
        plt.scatter(pixel_coords_map[:, 0, 0], pixel_coords_map[:, 0, 1], label='bot')
        plt.scatter(pixel_coords_map[:, (hypso_height // 2 - 1), 0], pixel_coords_map[:, (hypso_height // 2 - 1), 1],
                    label='mid')
        plt.scatter(pixel_coords_map[:, (hypso_height - 1), 0], pixel_coords_map[:, (hypso_height - 1), 1], label='top')
        # detector lines
        plt.scatter(pixel_coords_map[0, :, 0], pixel_coords_map[0, :, 1], label='first frame')
        plt.scatter(pixel_coords_map[477, :, 0], pixel_coords_map[477, :, 1], label='middle frame')
        plt.scatter(pixel_coords_map[955, :, 0], pixel_coords_map[955, :, 1], label='last frame')
        plt.grid()
        plt.legend()
        plt.axis('equal')
        plt.show()

    # Determining resample resolutions --------------------------------------------------------

    # time line x and y differences
    a = np.diff(pixel_coords_map[:, hypso_height // 2, 0])
    b = np.diff(pixel_coords_map[:, hypso_height // 2, 1])
    along_track_gsd = np.sqrt(a * a + b * b)
    along_track_mean_gsd = np.mean(along_track_gsd)

    # detector line x and y differences
    a = np.diff(pixel_coords_map[frame_count // 2, :, 0])
    b = np.diff(pixel_coords_map[frame_count // 2, :, 1])
    across_track_gsd = np.sqrt(a * a + b * b)
    across_track_mean_gsd = np.mean(across_track_gsd)

    # adjust resolutions
    resolution_scaling = 0.75  # lower value, higher res
    resolution_scaling_along = 0.9 * resolution_scaling
    resolution_scaling_across = 1.0 * resolution_scaling

    plt.figure()
    plt.plot(along_track_gsd, label=f'Along track {along_track_mean_gsd:.3f}')
    plt.plot(across_track_gsd,
             label=f'Across Track {across_track_mean_gsd:.2f}')
    fig = plt.title('Pixel-to-Pixel distances (GSD)')
    plt.grid()
    plt.legend()
    plt.xlabel('Pixel index')
    plt.ylabel('Distance [m]')
    plot_save_path = Path(geotiff_folder_path, capture_name + '-gsd.svg')
    plt.savefig(plot_save_path)

    # Finding bounding box and generating grid -----------------------------------------------------
    print('  Bounding boxes and resample grid ...')

    # printing input data corner coords of as lat lons and as map projection coordinates
    if DEBUG:
        print(pixels_lat[0, 0], pixels_lon[0, 0])
        print(pixels_lat[0, -1], pixels_lon[0, -1])
        print(pixels_lat[-1, 0], pixels_lon[-1, 0])
        print(pixels_lat[-1, -1], pixels_lon[-1, -1])
        print('')
        print(pixel_coords_map[0, 0, 0], pixel_coords_map[0, 0, 1])
        print(pixel_coords_map[0, -1, 0], pixel_coords_map[0, -1, 1])
        print(pixel_coords_map[-1, 0, 0], pixel_coords_map[-1, 0, 1])
        print(pixel_coords_map[-1, -1, 0], pixel_coords_map[-1, -1, 1])
        print('')

    pixel_coords_map_list = pixel_coords_map.reshape(
        frame_count * hypso_height, 2)
    bbox = [np.min(pixel_coords_map_list[:, 0]), np.max(pixel_coords_map_list[:, 0]), np.min(
        pixel_coords_map_list[:, 1]), np.max(pixel_coords_map_list[:, 1])]
    min_area_bbox = gref.minimum_bounding_rectangle(pixel_coords_map_list)
    # print(bbox)
    # print(min_area_bbox)

    grid_points, grid_dims = gref.gen_resample_grid(
        across_track_mean_gsd * resolution_scaling_across, along_track_mean_gsd * resolution_scaling_along, bbox)
    # print(grid_points.shape)
    # print(grid_points)

    grid_points_minimal, grid_dims_minimal = gref.gen_resample_grid_bbox_min(
        across_track_mean_gsd * resolution_scaling_across, along_track_mean_gsd * resolution_scaling_along,
        min_area_bbox)

    # TODO make the grid points appear in the same "order" (e.g. from top left to bottom right)
    # for both north aligned and minimal grid
    # print(grid_points[0,0,:])
    # print(grid_points_minimal[0,0,:])
    # print(grid_points[-1,-1,:])
    # print(grid_points_minimal[-1,-1,:])

    # print(grid_points_minimal[0,0,0], grid_points_minimal[0,0,1])
    # print(grid_points_minimal[0,-1,0], grid_points_minimal[0,-1,1])
    # print(grid_points_minimal[-1,0,0], grid_points_minimal[-1,0,1])
    # print(grid_points_minimal[-1,-1,0], grid_points_minimal[-1,-1,1])

    # plt.scatter(grid_points_minimal[::8,::8,0], grid_points_minimal[::8,::8,1], label='grid minimal')
    # plt.scatter(grid_points[::12,::12,0], grid_points[::12,::12,1], label='grid aligned')
    # plt.scatter(pixel_coords_map[::12,::12,0], pixel_coords_map[::12,::12,1], label='pixels')
    # plt.axis('equal')
    # plt.grid()
    # plt.show()

    grid_points_minimal.shape = (grid_dims_minimal[1] * grid_dims_minimal[0], 2)
    grid_points.shape = [grid_dims[1] * grid_dims[0], 2]

    # Get Grid points inside bounding polygon -------------------------------------------------------
    print('  Grid points inside bounding polygon ...')
    # contain mask to set some datapoints to zero
    contain_mask = boundingpath.contains_points(grid_points)
    print(
        f'    Points inside boundary: {np.sum(contain_mask)} / {grid_dims[1] * grid_dims[0]}')

    # Get Geo Transform ----------------------------------------------------------------------------
    angle = 180 * m.pi / 180.0
    # rotation = np.array([[m.cos(angle), m.sin(angle)],
    #                      [-m.sin(angle), m.cos(angle)]])
    # scaling = np.array([[across_track_mean_gsd * resolution_scaling_across, 0],
    #                     [0, -along_track_mean_gsd * resolution_scaling_along]])
    # affine_matrix = np.matmul(scaling, rotation)
    geotransform = (bbox[1], -across_track_mean_gsd * resolution_scaling_across,
                    0, bbox[2], 0, along_track_mean_gsd * resolution_scaling_along)

    # Registration, aka rectification, aka resampling, aka gridding -------------------------------------
    print('  Registration, aka rectification, aka resampling, aka gridding ...')
    grid_data_all_bands = np.zeros([grid_dims[1], grid_dims[0], band_count])
    resampling_method = 'nearest'  # 'linear' is slower

    geotiff_info = (
        grid_dims,
        [bbox[1], bbox[2]],
        [across_track_mean_gsd * resolution_scaling_across,
         along_track_mean_gsd * resolution_scaling_along],
        destination_epsg,
        geotiff_folder_path
    )

    threads_list = []
    for band_number in bands:
        # Multithreading in python info
        # https://www.tutorialspoint.com/python/python_multithreading.htm
        # https://realpython.com/intro-to-python-threading/

        args = (band_number, cube_data, pixel_coords_map_list, grid_points,
                resampling_method, contain_mask, geotiff_info, grid_data_all_bands)
        x = threading.Thread(target=interpolate_geotiff, args=args)
        x.start()
        threads_list.append(x)

    # Waiting until all threads are done
    for t in threads_list:
        t.join()

    return grid_dims, geotransform, destination_epsg, grid_data_all_bands, contain_mask

'''

'''
def interpolate_geotiff(band_number: int, cube_data: np.ndarray, pixel_coords_map_list: np.ndarray,
                        grid_points: np.ndarray, resampling_method: str, contain_mask: np.ndarray, geotiff_info: tuple,
                        grid_data_all_bands: np.ndarray) -> None:
    """
    Interplate Each band of the data cube before saving it to a GeoTiff

    :param band_number: Band number to interpolate
    :param cube_data: Numpy array of data cube
    :param pixel_coords_map_list:
    :param grid_points:
    :param resampling_method: Resampling method. Default is "nearest". Linear is better but slower.
    :param contain_mask:
    :param geotiff_info:
    :param grid_data_all_bands:

    :return: No return
    """
    print(f'      Starting band {band_number}')

    data_lines = cube_data.shape[0]
    data_samples = cube_data.shape[1]

    cube_data_single_band = cube_data[:, :, band_number]
    cube_data_single_band.shape = [data_lines * data_samples]

    grid_data_single_band = si.griddata(pixel_coords_map_list, cube_data_single_band, grid_points,
                                        method=resampling_method, rescale=False)

    grid_data_single_band[np.invert(contain_mask)] = 0
    grid_data_single_band.shape = [geotiff_info[0][1], geotiff_info[0][0]]
    grid_data_single_band = grid_data_single_band[:, ::-1]
    grid_data_all_bands[:, :, band_number] = grid_data_single_band

    if EXPORT_SINGLE_BANDS:  # For single band export does not export this band
        export_single_band_geotiff(Path(geotiff_info[4], f'band_{band_number}.tif'), grid_data_single_band,
                                   geotiff_info[0], geotiff_info[1], geotiff_info[2], geotiff_info[3])

    print(f'      Done with band {band_number}')
'''

'''
def export_single_band_geotiff(filename: Path, raster_data: np.ndarray, grid_dims: List[int], grid_origin: list,
                               grid_res: int, grid_epsg: int) -> None:
    """
    Export Single Band as individual channel GeoTiff

    :param filename: Absolute path to save GeoTiff
    :param raster_data: Numpy array of the data to save in the GeoTiff
    :param grid_dims:
    :param grid_origin:
    :param grid_res:
    :param grid_epsg:

    :return: No return.
    """
    dst_ds = gdal.GetDriverByName('GTiff').Create(
        str(filename), grid_dims[0], grid_dims[1], 1, gdal.GDT_Float64)
    geotransform = (grid_origin[0], -grid_res[0],
                    0.0, grid_origin[1], 0.0, grid_res[1])
    dst_ds.SetGeoTransform(geotransform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    srs.ImportFromEPSG(grid_epsg)
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
    # dst_ds.GetRasterBand(1).WriteArray(
    #     65535.0*raster_data / raster_data.max())   # write band to the raster
    dst_ds.GetRasterBand(1).WriteArray(
        raster_data)
    dst_ds.FlushCache()  # write to disk
'''