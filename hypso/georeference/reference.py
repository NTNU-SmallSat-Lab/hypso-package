import numpy as np
from importlib.resources import files
import os
from tqdm import tqdm
import csv
import pyproj as prj
import pandas as pd
# for mplpath.Path() and its contains_points() method
import matplotlib.path as mplpath
import shapely.geometry as sg
import matplotlib.pyplot as plt
from osgeo import gdal
import skimage
from osgeo import osr
# GIS
import cartopy.crs as ccrs
import math as m
import threading
import scipy.interpolate as si
from sklearn.preprocessing import PolynomialFeatures
import glob
# Custom library
# lib_path = os.path.join(os.path.dirname(sys.argv[0]), 'hsi-postprocessing/lib')
# sys.path.insert(0, lib_path)
# import georef as gref
from hypso.georeference import georef as gref

tiff_name = "_geotiff-full"

def start_coordinate_correction(top_folder_name: str, satinfo: dict, proj_metadata: dict,correction_type="lstsq"):
    point_file = glob.glob(top_folder_name + '/*.points')

    if len(point_file) == 0:
        print("Points File Was Not Found. No Correction done.")
        return satinfo["lat"], satinfo["lon"]
    else:
        print("Doing manual coordinate correction with .points file")
        lat, lon = coordinate_correction(
            point_file[0], proj_metadata,
            satinfo["lat"], satinfo["lon"],correction_type)

        return lat, lon


def coordinate_correction_matrix(filename, projection_metadata,correction_type):
    # Hypso CRS
    inproj = projection_metadata['inproj']
    # Convert WKT projection information into a cartopy projection
    projcs = inproj.GetAuthorityCode('PROJCS')
    projection_hypso = ccrs.epsg(projcs)
    projection_hypso_transformer = prj.Proj(projection_hypso)
    # Create projection transformer to go from QGIS to Hypso

    # Load GCP File
    gcps = None
    with open(filename,  encoding="utf8", errors='ignore') as csv_file:
        # Open CSV file
        reader = csv.reader(csv_file)

        next(reader, None)  # Skip row with CRS data
        # mapX, mapY, sourceX, sourceY, enable, dX, dY, residual
        column_names = list(next(reader, None))

        # Get CRS
        qgis_crs = 'epsg:3857'  # QGIS Also Uses as a Defaultb EPSG:4326 or EPSG:32617
        hypso_crs = projection_metadata["crs"]  # 'epsg:32632'

        # Transformer from dataset crs to destination crs
        transformer_src_dest = prj.Transformer.from_crs(
            qgis_crs, hypso_crs, always_xy=True)

        # Iterate through rows and
        for gcp in reader:

            if gcps is None:
                gcps = gcp
            else:
                gcps = np.row_stack((gcps, gcp))

        # Convert Data to Dataframe

        gcps = pd.DataFrame(gcps, columns=column_names)
        gcps.rename(columns={"mapX": "mapLon",
                             "mapY": "mapLat",
                             "sourceX": "uncorrectedHypsoLon",
                             "sourceY": "uncorrectedHypsoLat"}, inplace=True)

        # Add Columns for Map Results Convert from MapCRS to Hypso CRS

        gcps['transformed_mapLon'] = pd.Series(dtype='float')  # X is Longitude
        gcps['transformed_mapLat'] = pd.Series(dtype='float')  # Y is latitude

        # Iterate through each gcps array
        for index, row in gcps.iterrows():
            # Rows are copies, not references any more
            row = row.copy()
            # Transform Coordinates to Hypso CRS
            # inverse = False -> From QGIS to Hypso
            # inverse = True -> From Hypso to QGIS

            # Option 1: Assume that everything is already in coordinates
            transformed_lon, transformed_lat = projection_hypso_transformer(
                row['mapLon'], row['mapLat'], inverse=True)

            # Option 2: Assume that we need to move from one system to another
            # transformed_lon, transformed_lat = transformer_src_dest.transform(row['mapLon'], row['mapLat'])

            gcps.loc[index, 'transformed_mapLon'] = transformed_lon
            gcps.loc[index, 'transformed_mapLat'] = transformed_lat

            transformed_lon, transformed_lat = projection_hypso_transformer(row['uncorrectedHypsoLon'],
                                                                            row['uncorrectedHypsoLat'], inverse=True)
            gcps.loc[index, 'uncorrectedHypsoLon'] = transformed_lon
            gcps.loc[index, 'uncorrectedHypsoLat'] = transformed_lat

    # Estimate transform
    hypso_src = gcps[["uncorrectedHypsoLon",
                      "uncorrectedHypsoLat"]].to_numpy().astype(np.float32)
    hypso_dst = gcps[["transformed_mapLon",
                      "transformed_mapLat"]].to_numpy().astype(np.float32)

    print("Hypso Source\n",
          gcps[["uncorrectedHypsoLon", "uncorrectedHypsoLat"]])
    print("Hypso Destination\n",
          gcps[["transformed_mapLon", "transformed_mapLat"]])
    print('--------------------------------')

    # Get Affine Matrix
    # M = cv2.getAffineTransform(hypso_src[:3, :], hypso_dst[:3, :])  # Affine requires only 3 points

    # Estimate Affine 2D
    # M, mask = cv2.estimateAffine2D(hypso_src, hypso_dst, refineIters=50)
    # M, mask = cv2.estimateAffinePartial2D(
    # hypso_src, hypso_dst, refineIters=50)  # Good Results!

    M=None

    if correction_type=="polynomial":
        # 2nd Order skimage
        M = skimage.transform.PolynomialTransform()
        M.estimate(hypso_src, hypso_dst, order=3)


    elif correction_type=="homography": 
        M = skimage.transform.ProjectiveTransform()
        M.estimate(hypso_src, hypso_dst)

    elif correction_type=="affine": 
        # [[a0  a1  a2]
        # [b0  b1  b2]
        # [0   0    1]]
        M = skimage.transform.AffineTransform()
        M.estimate(hypso_src, hypso_dst)

    elif correction_type=="lstsq":
        # Using Numpy Least Squares ( Good Results!)
        M = []
        x = hypso_src[:, 0]
        y = hypso_dst[:, 0]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        M.append([m, c])
        x = hypso_src[:, 1]
        y = hypso_dst[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        M.append([m, c])

    return {correction_type:M}  # lat_coeff, lon_coeff


def coordinate_correction(point_file, projection_metadata, originalLat, originalLon, correction_type):
    # Use Utility File to Extract M
    M_dict = coordinate_correction_matrix(point_file, projection_metadata,correction_type)
    M_type=list(M_dict.keys())[0]
    M=M_dict[M_type]

    finalLat = originalLat.copy()
    finalLon = originalLon.copy()

    for i in range(originalLat.shape[0]):
        for j in range(originalLat.shape[1]):
            X = originalLon[i, j]
            Y = originalLat[i, j]

            modifiedLat = None
            modifiedLon=None

            # Second Degree Polynomial (Scikit)
            if M_type=="polynomial":
                # SK Image polynomial -----------------------------------------------

                lon_coeff = M.params[0]
                lat_coeff = M.params[1]

                deg = None
                if len(lon_coeff)==10:
                    deg=3
                elif len(lon_coeff)==6:
                    deg=2
                
                def apply_poly(X,Y,coeff,deg):
                    if deg==2:
                         return coeff[0] + coeff[1]*X + coeff[2]*Y + coeff[3]*X**2 + coeff[4]*X*Y + coeff[5]*Y**2 
                    elif deg==3:
                         return coeff[0] + coeff[1]*X + coeff[2]*Y + coeff[3]*X**2 + coeff[4]*X*Y + coeff[5]*Y**2 + coeff[6]*X**3 + coeff[7]*(X**2)*(Y) + coeff[8]*(X)*(Y**2) + coeff[9]*Y**3
                        
                modifiedLat= apply_poly(X,Y,lat_coeff,deg=deg)
                modifiedLon= apply_poly(X,Y,lon_coeff,deg=deg)

            elif M_type=="affine" or M_type=="homography":
                result=M.params @ np.array([[X],[Y],[1]]).ravel()
                modifiedLon=result[0]
                modifiedLat=result[1]

            # Np lin alg
            elif M_type=="lstsq":
                LonM = M[0]
                modifiedLon = LonM[0] * X + LonM[1]

                LatM = M[1]
                modifiedLat = LatM[0] * Y + LatM[1]


            finalLat[i, j] = modifiedLat
            finalLon[i, j] = modifiedLon

    return finalLat, finalLon


def array_to_geotiff(satObj, single_frame_array, file_name='custom_band'):
    if len(single_frame_array.shape) == 2:
        single_frame_array = np.expand_dims(single_frame_array, axis=2)

    # Setting some default values

    frame_count = satObj.info["frame_count"]
    hypso_height = satObj.info["image_height"]
    hypso_height_sensor = 1216

    hypso_width = 120
    hypso_width_sensor = 1936

    # HAndling arguments
    pixels_lat = satObj.info["lat"]
    pixels_lon = satObj.info["lon"]
    cube_path = satObj.info["top_folder_name"]

    # Setup
    pixels_lat.shape = (frame_count, hypso_height)
    pixels_lon.shape = (frame_count, hypso_height)

    dir_basename = str(os.path.basename(cube_path))

    output_path_tif = os.path.join(
        os.path.abspath(cube_path), dir_basename + tiff_name+'/'+dir_basename+'-'+file_name+'.tif')

    print('  Projecting pixel geodetic to map ...')
    bbox_geodetic = [np.min(pixels_lat), np.max(
        pixels_lat), np.min(pixels_lon), np.max(pixels_lon)]
    print('   ', bbox_geodetic)
    utm_crs_list = prj.database.query_utm_crs_info(datum_name="WGS 84",
                                                   area_of_interest=prj.aoi.AreaOfInterest(west_lon_degree=bbox_geodetic[2], south_lat_degree=bbox_geodetic[0], east_lon_degree=bbox_geodetic[3], north_lat_degree=bbox_geodetic[1],))
    print(f'    using UTM map: ' +
          utm_crs_list[0].name, 'EPSG:', utm_crs_list[0].code)

    # crs_25832 = prj.CRS.from_epsg(25832) # UTM32N
    # crs_32717 = prj.CRS.from_epsg(32717) # UTM17S
    crs_4326 = prj.CRS.from_epsg(4326)  # Unprojected [(lat,lon), probably]
    source_crs = crs_4326
    destination_epsg = int(utm_crs_list[0].code)
    destination_crs = prj.CRS.from_epsg(destination_epsg)
    # latlon_to_proj.transform(lat,lon) returns (east, north)
    latlon_to_proj = prj.Transformer.from_crs(source_crs, destination_crs)
    # proj_to_latlon = prj.Transformer.from_crs(destination_crs, source_crs)

    pixel_coords_map = np.zeros([frame_count, hypso_height, 2])
    for i in range(frame_count):
        for j in range(hypso_height):
            pixel_coords_map[i, j, :] = latlon_to_proj.transform(
                pixels_lat[i, j], pixels_lon[i, j])

    dg_bounding_path = np.concatenate(
        (pixel_coords_map[:, 0, :], pixel_coords_map[::-1, (hypso_height-1), :]))
    # this is the Direct Georeferencing outline
    boundingpath = mplpath.Path(dg_bounding_path)

    boundingpath_area = sg.Polygon(dg_bounding_path).area

    # time line x and y differences
    a = np.diff(pixel_coords_map[:, hypso_height//2, 0])
    b = np.diff(pixel_coords_map[:, hypso_height//2, 1])
    along_track_gsd = np.sqrt(a*a + b*b)
    along_track_mean_gsd = np.mean(along_track_gsd)

    # detector line x and y differences
    a = np.diff(pixel_coords_map[frame_count//2, :, 0])
    b = np.diff(pixel_coords_map[frame_count//2, :, 1])
    across_track_gsd = np.sqrt(a*a + b*b)
    across_track_mean_gsd = np.mean(across_track_gsd)

    # adjust resolutions
    resolution_scaling = 0.75  # lower value, higher res
    resolution_scaling_along = 0.9*resolution_scaling
    resolution_scaling_across = 1.0*resolution_scaling

    pixel_coords_map_list = pixel_coords_map.reshape(
        frame_count*hypso_height, 2)
    bbox = [np.min(pixel_coords_map_list[:, 0]), np.max(pixel_coords_map_list[:, 0]), np.min(
        pixel_coords_map_list[:, 1]), np.max(pixel_coords_map_list[:, 1])]
    min_area_bbox = gref.minimum_bounding_rectangle(pixel_coords_map_list)
    # print(bbox)
    # print(min_area_bbox)

    grid_points, grid_dims = gref.gen_resample_grid(
        across_track_mean_gsd*resolution_scaling_across, along_track_mean_gsd*resolution_scaling_along, bbox)
    # print(grid_points.shape)
    # print(grid_points)

    grid_points_minimal, grid_dims_minimal = gref.gen_resample_grid_bbox_min(
        across_track_mean_gsd*resolution_scaling_across, along_track_mean_gsd*resolution_scaling_along, min_area_bbox)

    grid_points_minimal.shape = (grid_dims_minimal[1]*grid_dims_minimal[0], 2)
    grid_points.shape = [grid_dims[1]*grid_dims[0], 2]

    print('  Grid points inside bounding polygon ...')
    # contain mask to set some datapoints to zero
    contain_mask = boundingpath.contains_points(grid_points)
    print(
        f'    Points inside boundary: {np.sum(contain_mask)} / {grid_dims[1]*grid_dims[0]}')

    print('  Registration, aka rectification, aka resampling, aka gridding ...')

    band_count = 1  # Single frame
    cube_data = single_frame_array

    bands = [0]  # Single frame

    grid_data_all_bands = np.zeros([grid_dims[1], grid_dims[0], band_count])

    resampling_method = 'nearest'

    geotiff_info = (
        grid_dims,
        [bbox[1], bbox[2]],
        [across_track_mean_gsd*resolution_scaling_across,
            along_track_mean_gsd*resolution_scaling_along],
        destination_epsg,
        output_path_tif
    )
    threads_list = []

    # Multithreading in python info
    # https://www.tutorialspoint.com/python/python_multithreading.htm
    # https://realpython.com/intro-to-python-threading/

    for band_number in bands:
        args = (band_number, cube_data, pixel_coords_map_list, grid_points,
                resampling_method, contain_mask, geotiff_info, grid_data_all_bands)
        x = threading.Thread(target=interpolate_geotiff, args=args)
        x.start()
        threads_list.append(x)

    # Waiting until all threads are done
    for t in threads_list:
        t.join()

    dst_ds_single = gdal.GetDriverByName('GTiff').Create(
        output_path_tif, grid_dims[0], grid_dims[1], 2, gdal.GDT_Byte)  # For Classes and Alpha

    angle = -10*m.pi/180.0
    rotation = np.array([[m.cos(angle), m.sin(angle)],
                        [-m.sin(angle), m.cos(angle)]])
    scaling = np.array([[across_track_mean_gsd*resolution_scaling_across, 0],
                       [0, -along_track_mean_gsd*resolution_scaling_along]])
    affine_matrix = np.matmul(scaling, rotation)
    geotransform = (grid_points_minimal[0, 0], affine_matrix[0, 0], affine_matrix[0, 1],
                    grid_points_minimal[0, 1], affine_matrix[1, 0], affine_matrix[1, 1])

    # dst_ds_minimal.SetGeoTransform(geotransform)    # specify coords
    # srs = osr.SpatialReference()            # establish encoding
    # srs.ImportFromEPSG(destination_epsg)
    # dst_ds_minimal.SetProjection(srs.ExportToWkt()) # export coords to file
    # for i, rgb_band_index in enumerate(rgb_band_indices):
    #    dst_ds_minimal.GetRasterBand(i+1).WriteArray(grid_data_all_bands_minimal[:,:,rgb_band_index])
    # dst_ds_minimal.FlushCache()                     # write to disk

    angle = 180*m.pi/180.0
    rotation = np.array([[m.cos(angle), m.sin(angle)],
                        [-m.sin(angle), m.cos(angle)]])
    scaling = np.array([[across_track_mean_gsd*resolution_scaling_across, 0],
                       [0, -along_track_mean_gsd*resolution_scaling_along]])
    affine_matrix = np.matmul(scaling, rotation)
    # geotransform = (bbox[0], affine_matrix[0,0], affine_matrix[0,1], bbox[3], affine_matrix[1,0], affine_matrix[1,1])
    geotransform = (bbox[1], -across_track_mean_gsd*resolution_scaling_across,
                    0, bbox[2], 0, along_track_mean_gsd*resolution_scaling_along)

    # RGBA -------------------------------------------------------
    # Alpha mask for transparency in the RGBA geotiff
    alpha_mask = np.zeros([grid_dims[1], grid_dims[0]])
    alpha_mask.shape = [grid_dims[1]*grid_dims[0]]
    alpha_mask[contain_mask] = 255
    alpha_mask.shape = [grid_dims[1], grid_dims[0]]
    alpha_mask = alpha_mask[:, ::-1]

    dst_ds_single.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(destination_epsg)
    dst_ds_single.SetProjection(
        srs.ExportToWkt())  # export coords to file

    dst_ds_single.GetRasterBand(1).WriteArray(
        255.0*grid_data_all_bands[:, :, 0])

    dst_ds_single.GetRasterBand(2).WriteArray(alpha_mask)
    dst_ds_single.FlushCache()                  # write to disk


def generate_geotiff(satObj):
    print('Generating Geotiff ************************************')
    # print('  This script requires three arguments:')
    # print('    1. path to a pixels latitude numpy file')
    # print('    2. path to a pixels longitude numpy file')
    # print('    3. path to a decompressed hypso cube file')
    # print('  Followed by up to six optional arguments in order:')
    # print('    4. frames/lines in the cube (default 956)')
    # print('    5. pixels/samples in a frame/line (default 684)')
    # print('    6. bands in a pixel/sample (default 120)')
    # print('    7. r band index (default 75 for 650nm)')
    # print('    8. g band index (default 46 for 550nm)')
    # print('    9. b band index (default 18 for 450nm)')

    # Setting some default values

    frame_count = satObj.info["frame_count"]
    hypso_height = satObj.info["image_height"]
    hypso_height_sensor = 1216

    hypso_width = 120
    hypso_width_sensor = 1936

    r_band_index = 75  # Old 59
    g_band_index = 46  # Old 70
    b_band_index = 18  # Old 89

    # HAndling arguments
    pixels_lat = satObj.info["lat"]
    pixels_lon = satObj.info["lon"]
    cube_path = satObj.info["top_folder_name"]

    # Setup
    pixels_lat.shape = (frame_count, hypso_height)
    pixels_lon.shape = (frame_count, hypso_height)

    dir_basename = str(os.path.basename(cube_path))


    geotiff_folder_path = os.path.join(os.path.abspath(
        cube_path), dir_basename + tiff_name+'/')
    output_path_band_tif_base = os.path.join(
        os.path.abspath(cube_path), dir_basename + tiff_name+'/band_')
    output_path_rgb_tif = os.path.join(
        os.path.abspath(cube_path), dir_basename + tiff_name+'/'+dir_basename+'-rgb.tif')
    output_path_rgba_tif = os.path.join(
        os.path.abspath(cube_path), dir_basename + tiff_name+'/'+dir_basename+'-rgba_8bit.tif')
    output_path_full_tif = os.path.join(
        os.path.abspath(cube_path), dir_basename + tiff_name+'/'+dir_basename+'-full.tif')

    output_path_geo_info = os.path.join(
        os.path.abspath(cube_path), dir_basename + tiff_name+'/'+dir_basename+'-geometric-meta-info-full.txt')

    print('  Projecting pixel geodetic to map ...')
    bbox_geodetic = [np.min(pixels_lat), np.max(
        pixels_lat), np.min(pixels_lon), np.max(pixels_lon)]
    print('   ', bbox_geodetic)
    utm_crs_list = prj.database.query_utm_crs_info(datum_name="WGS 84",
                                                   area_of_interest=prj.aoi.AreaOfInterest(west_lon_degree=bbox_geodetic[2], south_lat_degree=bbox_geodetic[0], east_lon_degree=bbox_geodetic[3], north_lat_degree=bbox_geodetic[1],))
    print(f'    using UTM map: ' +
          utm_crs_list[0].name, 'EPSG:', utm_crs_list[0].code)

    # crs_25832 = prj.CRS.from_epsg(25832) # UTM32N
    # crs_32717 = prj.CRS.from_epsg(32717) # UTM17S
    crs_4326 = prj.CRS.from_epsg(4326)  # Unprojected [(lat,lon), probably]
    source_crs = crs_4326
    destination_epsg = int(utm_crs_list[0].code)
    destination_crs = prj.CRS.from_epsg(destination_epsg)
    # latlon_to_proj.transform(lat,lon) returns (east, north)
    latlon_to_proj = prj.Transformer.from_crs(source_crs, destination_crs)
    # proj_to_latlon = prj.Transformer.from_crs(destination_crs, source_crs)

    pixel_coords_map = np.zeros([frame_count, hypso_height, 2])
    for i in range(frame_count):
        for j in range(hypso_height):
            pixel_coords_map[i, j, :] = latlon_to_proj.transform(
                pixels_lat[i, j], pixels_lon[i, j])

    dg_bounding_path = np.concatenate(
        (pixel_coords_map[:, 0, :], pixel_coords_map[::-1, (hypso_height-1), :]))
    # this is the Direct Georeferencing outline
    boundingpath = mplpath.Path(dg_bounding_path)

    boundingpath_area = sg.Polygon(dg_bounding_path).area

    # Make Dir To export
    isExist = os.path.exists(os.path.join(
        os.path.abspath(cube_path), dir_basename + tiff_name+'/'))
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(os.path.join(
            os.path.abspath(cube_path), dir_basename + tiff_name+'/'))

    with open(output_path_geo_info, 'a') as f:
        f.write(
            f'Imaged area (square kilometers): {boundingpath_area/1000000.0:09.5f}\n')
    print(
        f'    Area covered by image: {boundingpath_area} m^2 = {boundingpath_area/1000000.0} km^2')

    # plt.scatter(pixels_lon[:,0], pixels_lat[:,0], label='bot')
    # plt.scatter(pixels_lon[:,(hypso_height//2-1)], pixels_lat[:,(hypso_height//2-1)], label='mid')
    # plt.scatter(pixels_lon[:,(hypso_height-1)], pixels_lat[:,(hypso_height-1)], label='top')
    # plt.scatter(pixels_lon[0,:], pixels_lat[0,:], label='first frame')
    # plt.scatter(pixels_lon[955,:], pixels_lat[955,:], label='last frame')
    # plt.grid()
    # plt.legend()
    # plt.show()
    # # time lines
    # plt.scatter(pixel_coords_map[:,0,0], pixel_coords_map[:,0,1], label='bot')
    # plt.scatter(pixel_coords_map[:,(hypso_height//2-1),0], pixel_coords_map[:,(hypso_height//2-1),1], label='mid')
    # plt.scatter(pixel_coords_map[:,(hypso_height-1),0], pixel_coords_map[:,(hypso_height-1),1], label='top')
    # # detector lines
    # plt.scatter(pixel_coords_map[0,:,0], pixel_coords_map[0,:,1], label='first frame')
    # plt.scatter(pixel_coords_map[477,:,0], pixel_coords_map[477,:,1], label='middle frame')
    # plt.scatter(pixel_coords_map[955,:,0], pixel_coords_map[955,:,1], label='last frame')
    # plt.grid()
    # plt.legend()
    # plt.axis('equal')
    # plt.show()

    # Determining resample resolutions

    # time line x and y differences
    a = np.diff(pixel_coords_map[:, hypso_height//2, 0])
    b = np.diff(pixel_coords_map[:, hypso_height//2, 1])
    along_track_gsd = np.sqrt(a*a + b*b)
    along_track_mean_gsd = np.mean(along_track_gsd)

    # detector line x and y differences
    a = np.diff(pixel_coords_map[frame_count//2, :, 0])
    b = np.diff(pixel_coords_map[frame_count//2, :, 1])
    across_track_gsd = np.sqrt(a*a + b*b)
    across_track_mean_gsd = np.mean(across_track_gsd)

    # adjust resolutions
    resolution_scaling = 0.75  # lower value, higher res
    resolution_scaling_along = 0.9*resolution_scaling
    resolution_scaling_across = 1.0*resolution_scaling

    plt.plot(along_track_gsd, label=f'Along track {along_track_mean_gsd:.3f}')
    plt.plot(across_track_gsd,
             label=f'Across Track {across_track_mean_gsd:.2f}')
    plt.title('Pixel-to-Pixel distances (GSD)')
    plt.grid()
    plt.legend()
    plt.xlabel('Pixel index')
    plt.ylabel('Distance [m]')
    plot_save_path = os.path.join(
        os.path.abspath(cube_path), dir_basename + tiff_name+'/'+dir_basename+'-gsd.svg')
    plt.savefig(plot_save_path)
    # plt.show()

    # exit(1)

    # Finding bounding box and generating grid
    print('  Bounding boxes and resample grid ...')

    # printing input data corner coords of as lat lons and as map projection coordinates
    # print(pixels_lat[0,0], pixels_lon[0,0])
    # print(pixels_lat[0,-1], pixels_lon[0,-1])
    # print(pixels_lat[-1,0], pixels_lon[-1,0])
    # print(pixels_lat[-1,-1], pixels_lon[-1,-1])
    # print('')
    # print(pixel_coords_map[0,0,0], pixel_coords_map[0,0,1])
    # print(pixel_coords_map[0,-1,0], pixel_coords_map[0,-1,1])
    # print(pixel_coords_map[-1,0,0], pixel_coords_map[-1,0,1])
    # print(pixel_coords_map[-1,-1,0], pixel_coords_map[-1,-1,1])
    # print('')

    pixel_coords_map_list = pixel_coords_map.reshape(
        frame_count*hypso_height, 2)
    bbox = [np.min(pixel_coords_map_list[:, 0]), np.max(pixel_coords_map_list[:, 0]), np.min(
        pixel_coords_map_list[:, 1]), np.max(pixel_coords_map_list[:, 1])]
    min_area_bbox = gref.minimum_bounding_rectangle(pixel_coords_map_list)
    # print(bbox)
    # print(min_area_bbox)

    grid_points, grid_dims = gref.gen_resample_grid(
        across_track_mean_gsd*resolution_scaling_across, along_track_mean_gsd*resolution_scaling_along, bbox)
    # print(grid_points.shape)
    # print(grid_points)

    grid_points_minimal, grid_dims_minimal = gref.gen_resample_grid_bbox_min(
        across_track_mean_gsd*resolution_scaling_across, along_track_mean_gsd*resolution_scaling_along, min_area_bbox)

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

    grid_points_minimal.shape = (grid_dims_minimal[1]*grid_dims_minimal[0], 2)
    grid_points.shape = [grid_dims[1]*grid_dims[0], 2]

    print('  Grid points inside bounding polygon ...')
    # contain mask to set some datapoints to zero
    contain_mask = boundingpath.contains_points(grid_points)
    print(
        f'    Points inside boundary: {np.sum(contain_mask)} / {grid_dims[1]*grid_dims[0]}')

    print('  Registration, aka rectification, aka resampling, aka gridding ...')

    band_count = hypso_width
    cube_data = satObj.l1b_cube  # could be rawcube
    # cube_data=np.fromfile(cube_path, dtype='uint16')
    # cube_data.shape = [frame_count, hypso_height, hypso_width]

    try:
        os.mkdir(geotiff_folder_path)
    except FileExistsError:
        pass

    extra_band_index = (g_band_index-4)//4

    # bands = [extra_band_index, r_band_index, g_band_index, b_band_index]
    bands = [i for i in range(120)]
    rgb_band_indices = [r_band_index, g_band_index, b_band_index]
    grid_data_all_bands = np.zeros([grid_dims[1], grid_dims[0], band_count])
    grid_data_all_bands_minimal = np.zeros(
        [grid_dims_minimal[1], grid_dims_minimal[0], band_count])
    resampling_method = 'nearest'
    #resampling_method = 'linear'

    geotiff_info = (
        grid_dims,
        [bbox[1], bbox[2]],
        [across_track_mean_gsd*resolution_scaling_across,
            along_track_mean_gsd*resolution_scaling_along],
        destination_epsg,
        output_path_band_tif_base
    )
    MULTI_THREAD=True
    if MULTI_THREAD:
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
    else:
        for ii in tqdm (range (len(bands)), desc="Loading Bands..."):
            band_number=bands[ii]
            interpolate_geotiff(band_number, cube_data, pixel_coords_map_list, grid_points,
                    resampling_method, contain_mask, geotiff_info, grid_data_all_bands)

    max_value_rgb = 0.0
    for rgb_band_index in rgb_band_indices:
        max_current_band = grid_data_all_bands[:, :, rgb_band_index].max()
        if max_current_band > max_value_rgb:
            max_value_rgb = max_current_band

    # grid_data_single_band_minimal = grid_data_single_band_minimal[:,::-1]

    # plt.imshow(grid_data_single_band)
    # plt.colorbar()
    # plt.show()

    # plt.imshow(cube_data_single_band)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(grid_data_single_band_minimal[:,::-1])
    # plt.colorbar()
    # plt.show()

    # some Geotiff reference info:
    # https://gis.stackexchange.com/questions/380607/how-to-geo-reference-a-tif-image-knowing-corner-coordinates
    # https://gis.stackexchange.com/questions/275125/expressing-raster-rotation-with-gdal-geotransform-python

    # dst_ds_minimal = gdal.GetDriverByName('GTiff').Create('myGeoTIFF_minimal.tif', grid_dims_minimal[0], grid_dims_minimal[1], 3, gdal.GDT_UInt16)
    dst_ds = gdal.GetDriverByName('GTiff').Create(
        output_path_rgb_tif, grid_dims[0], grid_dims[1], 3, gdal.GDT_UInt16)
    dst_ds_alpha_channel = gdal.GetDriverByName('GTiff').Create(
        output_path_rgba_tif, grid_dims[0], grid_dims[1], 4, gdal.GDT_Byte)

    dst_ds_full = gdal.GetDriverByName('GTiff').Create(
        output_path_full_tif, grid_dims[0], grid_dims[1], 120, gdal.GDT_Float64)

    # xmin, ymin, xmax, ymax = [min(grid_points_minimal[:,0]), min(grid_points_minimal[:,1]), max(grid_points_minimal[:,0]), max(grid_points_minimal[:,1])]
    # xres = (xmax - xmin) / float(grid_dims_minimal[0])
    # yres = (ymax - ymin) / float(grid_dims_minimal[1])
    # angle = -10*m.pi/180.0
    # rotation = np.array([[m.cos(angle), m.sin(angle)],[-m.sin(angle), m.cos(angle)]])
    # scaling = np.array([[xres, 0],[0, -yres]])
    # affine_matrix = np.matmul(scaling, rotation)
    # geotransform = (xmin, affine_matrix[0,0], affine_matrix[0,1], ymax, affine_matrix[1,0], affine_matrix[1,1])

    # geotransform = (xmin, xres, 0, ymax, 0, -yres)

    # geotransform = (grid_points_minimal[0,0], across_track_mean_gsd*resolution_scaling_across, 0, grid_points_minimal[0,1], 0, -along_track_mean_gsd*resolution_scaling_along)

    angle = -10*m.pi/180.0
    rotation = np.array([[m.cos(angle), m.sin(angle)],
                        [-m.sin(angle), m.cos(angle)]])
    scaling = np.array([[across_track_mean_gsd*resolution_scaling_across, 0],
                       [0, -along_track_mean_gsd*resolution_scaling_along]])
    affine_matrix = np.matmul(scaling, rotation)
    geotransform = (grid_points_minimal[0, 0], affine_matrix[0, 0], affine_matrix[0, 1],
                    grid_points_minimal[0, 1], affine_matrix[1, 0], affine_matrix[1, 1])

    # dst_ds_minimal.SetGeoTransform(geotransform)    # specify coords
    # srs = osr.SpatialReference()            # establish encoding
    # srs.ImportFromEPSG(destination_epsg)
    # dst_ds_minimal.SetProjection(srs.ExportToWkt()) # export coords to file
    # for i, rgb_band_index in enumerate(rgb_band_indices):
    #    dst_ds_minimal.GetRasterBand(i+1).WriteArray(grid_data_all_bands_minimal[:,:,rgb_band_index])
    # dst_ds_minimal.FlushCache()                     # write to disk

    angle = 180*m.pi/180.0
    rotation = np.array([[m.cos(angle), m.sin(angle)],
                        [-m.sin(angle), m.cos(angle)]])
    scaling = np.array([[across_track_mean_gsd*resolution_scaling_across, 0],
                       [0, -along_track_mean_gsd*resolution_scaling_along]])
    affine_matrix = np.matmul(scaling, rotation)
    # geotransform = (bbox[0], affine_matrix[0,0], affine_matrix[0,1], bbox[3], affine_matrix[1,0], affine_matrix[1,1])
    geotransform = (bbox[1], -across_track_mean_gsd*resolution_scaling_across,
                    0, bbox[2], 0, along_track_mean_gsd*resolution_scaling_along)

    # RGB ------------------------------------------------------
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(destination_epsg)
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
    for i, rgb_band_index in enumerate(rgb_band_indices):
        dst_ds.GetRasterBand(
            i+1).WriteArray(grid_data_all_bands[:, :, rgb_band_index])
    dst_ds.FlushCache()                  # write to disk

    # RGBA -------------------------------------------------------
    # Alpha mask for transparency in the RGBA geotiff
    alpha_mask = np.zeros([grid_dims[1], grid_dims[0]])
    alpha_mask.shape = [grid_dims[1]*grid_dims[0]]
    alpha_mask[contain_mask] = 255
    alpha_mask.shape = [grid_dims[1], grid_dims[0]]
    alpha_mask = alpha_mask[:, ::-1]

    dst_ds_alpha_channel.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(destination_epsg)
    dst_ds_alpha_channel.SetProjection(
        srs.ExportToWkt())  # export coords to file

    for i, rgb_band_index in enumerate(rgb_band_indices):
        dst_ds_alpha_channel.GetRasterBand(
            i+1).WriteArray(255.0*grid_data_all_bands[:, :, rgb_band_index] / max_value_rgb)

    dst_ds_alpha_channel.GetRasterBand(4).WriteArray(alpha_mask)
    dst_ds_alpha_channel.FlushCache()                  # write to disk

    # Full Band Tiff --------------------------------
    dst_ds_full.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(destination_epsg)
    dst_ds_full.SetProjection(
        srs.ExportToWkt())  # export coords to file

    for i, band_index in enumerate(bands):
        dst_ds_full.GetRasterBand(
            i+1).WriteArray(grid_data_all_bands[:, :, band_index])

    dst_ds_full.FlushCache()                  # write to disk

    print('Done')


def interpolate_geotiff(band_number, cube_data, pixel_coords_map_list, grid_points,
                        resampling_method, contain_mask, geotiff_info, grid_data_all_bands):

    #print(f'      Starting band {band_number}')

    data_lines = cube_data.shape[0]
    data_samples = cube_data.shape[1]

    cube_data_single_band = cube_data[:, :, band_number]
    cube_data_single_band.shape = [data_lines * data_samples]


    grid_data_single_band = si.griddata(pixel_coords_map_list, cube_data_single_band, grid_points,
                                        method=resampling_method, rescale=False)

    
    # cube_data_single_band.shape = [data_lines, data_samples]

    # grid_data_single_band_minimal = si.griddata(pixel_coords_map_list, cube_data_single_band, grid_points_minimal, method=resampling_method, rescale=False)
    # grid_data_single_band_minimal.shape = [grid_dims_minimal[1], grid_dims_minimal[0]]
    # grid_data_all_bands_minimal[:,:,band_number] = grid_data_single_band_minimal

     
    grid_data_single_band[np.invert(contain_mask)] = 0
    grid_data_single_band.shape = [geotiff_info[0][1], geotiff_info[0][0]]
    grid_data_single_band = grid_data_single_band[:, ::-1]
    grid_data_all_bands[:, :, band_number] = grid_data_single_band



    if cube_data.shape[2] != 1:  # For single band export does not export this band
        # export_single_band_geotiff(filename, raster_data, grid_dims, grid_res, grid_epsg):
        # export_single_band_geotiff(f'{geotiff_info[4]}{band_number}.tif', grid_data_single_band,
        #                            geotiff_info[0], geotiff_info[1], geotiff_info[2], geotiff_info[3])
        export_single_band_geotiff(f'{geotiff_info[4]}{band_number}.tif', grid_data_single_band,
                                   geotiff_info[0], geotiff_info[1], geotiff_info[2], geotiff_info[3])
        
    print(f'      Done with band {band_number}')


def export_single_band_geotiff(filename, raster_data, grid_dims, grid_origin, grid_res, grid_epsg):
    dst_ds = gdal.GetDriverByName('GTiff').Create(
        filename, grid_dims[0], grid_dims[1], 1, gdal.GDT_UInt16)
    geotransform = (grid_origin[0], -grid_res[0],
                    0.0, grid_origin[1], 0.0, grid_res[1])
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(grid_epsg)
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(
        65535.0*raster_data / raster_data.max())   # write band to the raster
    dst_ds.FlushCache()                  # write to disk
