'''Calculates destriping correction matrix from in-orbit data.

2023-03-20 MBH'''

import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import interp1d

# Local files
import sys
sys.path.insert(0, '../Tools')
import utilities as util
import read_images
import calibration_pipeline_functions as cal_func
import show_bip

figsize1 = 8
figsize2 = 4
plt.rcParams['figure.figsize'] = [figsize1, figsize2]
plt.rcParams.update({'font.size': 11}) 
color_map = plt.cm.get_cmap('YlGnBu') 
color_map = util.truncate_colormap(color_map, 0.1, 1)


def main():

    # get_destriping_correction_frame_for_684spatial_120bands()
    # get_destriping_correction_frame_for_684spatial_1080bands()

    # Plot destriping correction matrix
    coeff_path = 'Coefficients/'

    # Spatial 684, spectral 120 (AOI, bin = 9) 
    destriping_correction_matrix_684spatial_120bands = util.readCsvFile(coeff_path+'destriping_correction_matrix_HYPSO-1_684spatial_120bands.csv')
    fig, ax = plt.subplots()
    plt.imshow(destriping_correction_matrix_684spatial_120bands, aspect=0.1, vmin=-1, vmax=1)
    
    # Spatial 684, spectral 1080 (AOI, no binning)
    destriping_correction_matrix_684spatial_1080bands = util.readCsvFile(coeff_path+'destriping_correction_matrix_HYPSO-1_684spatial_1080bands.csv')
    fig, ax = plt.subplots()
    plt.imshow(destriping_correction_matrix_684spatial_1080bands, vmin=-2, vmax=2)

    plt.show()

    return None


def get_destriping_correction_frame_for_684spatial_120bands():

    ## Data
    path = '../../../Data/HYPSO-1/frohavet/'
    filename = path + '2022-05-17/frohavet-2022-05-17.bip' 
    # Ini file
    frame_count = 956
    exposure = 49.7349
    fps = 20
    row_count = 684
    column_count = 1080
    sample_divisor = 1
    bin_factor = 9
    aoi_x = 428
    aoi_y = 266

    # Metadata for script
    x_start = aoi_x
    x_stop = aoi_x + column_count 
    y_start = aoi_y
    y_stop = aoi_y + row_count
    bin_x = bin_factor
    image_height = int(row_count)
    image_width = int(column_count/bin_factor)
    im_size = image_height*image_width
    num_frames = frame_count
    exp = exposure/1000

    # Calibration coefficients
    coeff_path = 'Coefficients/'
    spectral_band_matrix = util.readCsvFile(coeff_path+'spectral_calibration_matrix_HYPSO-1.csv')
    radiometric_calibration_matrix = util.readCsvFile(coeff_path+'radiometric_calibration_matrix_HYPSO-1.csv')
    background_value = 8 # TODO: can improve this one with adjusted values for exp/temp

    # Read cube
    cube = read_images.read_bip_cube(filename, image_width, image_height)
    cube = cube[:,:,::-1] # Flip HYPSO-1 cube

    # Adjust calibration matrices to cropped and binned frames

    # Scale to get 12-bit values
    # The values are summed when saving binned cubes, so divide by binned 
    # values should provide the original 12-bit counts
    cube = cube/bin_x

    # Crop and bin spectral calibration matrix
    spectral_band_matrix = cal_func.crop_and_bin_matrix(spectral_band_matrix, x_start, x_stop, y_start, y_stop, bin_x)

    # Crop and bin radiometric calibration matrix
    # Assumes that the cube is scaled correctly to 12-bit values 
    radiometric_calibration_matrix = cal_func.crop_and_bin_matrix(radiometric_calibration_matrix, x_start, x_stop, y_start, y_stop, bin_x)

    # Radiometric calibration
    cube_calibrated = cal_func.calibrate_cube(cube, exp, radiometric_calibration_matrix, background_value)

    # Smile correction
    cube_calibrated_smile_corrected = cal_func.smile_correction_cube(cube_calibrated, spectral_band_matrix)

    # Destriping
    water_mask = cal_func.make_mask(cube, sat_val_scale=0.25)
    destriping_correction_matrix_calibrated = cal_func.get_destriping_correction_matrix(cube_calibrated_smile_corrected, water_mask)

    # Save radiometric calibration matrix
    np.savetxt('Coefficients/destriping_correction_matrix_HYPSO-1_684spatial_120bands.csv', destriping_correction_matrix_calibrated, delimiter=",") 

    return None


def get_destriping_correction_frame_for_684spatial_1080bands(plot=False):

    ## FLORIDA
    # Some features in the ocean, but checking it out after all
    # Path and filename of raw cube
    path = '../../../Data/HYPSO-1/uniform/'
    filename = path + '20221209_CaptureDL_florida_2022_12_06T15_59_55/florida_2022-12-06.bip' 

    # Metadata from capture_config.ini
    flags = 0x00000201
    camera_ID = 2
    frame_count = 106
    exposure = 30.0063
    fps = 12
    row_count = 684
    column_count = 1080
    sample_divisor = 1
    bin_factor = 1
    aoi_x = 428
    aoi_y = 266
    gain = 0
    temp_log_period_ms = 10000

    # Metadata for script
    x_start = aoi_x
    x_stop = aoi_x + column_count 
    y_start = aoi_y
    y_stop = aoi_y + row_count
    bin_x = bin_factor
    image_height = int(row_count)
    image_width = int(column_count/bin_factor)
    im_size = image_height*image_width
    num_frames = frame_count
    exp = exposure/1000

    # Calibration coefficients
    coeff_path = 'Coefficients/'
    spectral_band_matrix = util.readCsvFile(coeff_path+'spectral_calibration_matrix_HYPSO-1.csv')
    radiometric_calibration_matrix = util.readCsvFile(coeff_path+'radiometric_calibration_matrix_HYPSO-1.csv')
    background_value = 8 # TODO: can improve this one with adjusted values for exp/temp

    # Read cube
    cube = read_images.read_bip_cube(filename, image_width, image_height)
    cube = cube[:,:,::-1] # Flip HYPSO-1 cube
    if plot:
        show_bip.show_bip_cube(cube, 500, 510, image_height, num_frames, 'Cube')

    # Adjust calibration matrices to cropped and binned frames

    # Scale to get 12-bit values
    # The values are summed when saving binned cubes, so divide by binned 
    # values should provide the original 12-bit counts
    cube = cube/bin_x

    # Crop and bin spectral calibration matrix
    spectral_band_matrix = cal_func.crop_and_bin_matrix(spectral_band_matrix, x_start, x_stop, y_start, y_stop, bin_x)

    # Crop and bin radiometric calibration matrix
    # Assumes that the cube is scaled correctly to 12-bit values 
    radiometric_calibration_matrix = cal_func.crop_and_bin_matrix(radiometric_calibration_matrix, x_start, x_stop, y_start, y_stop, bin_x)

    # Radiometric calibration
    cube_calibrated = cal_func.calibrate_cube(cube, exp, radiometric_calibration_matrix, background_value)
    if plot:
        show_bip.show_bip_cube(cube_calibrated, 500, 510, image_height, num_frames, 'Calibrated', calibrated=True)

    # Smile correction
    cube_calibrated_smile_corrected = cal_func.smile_correction_cube(cube_calibrated, spectral_band_matrix)
    if plot:
        show_bip.show_bip_cube(cube_calibrated_smile_corrected, 500, 510, image_height, num_frames, 'Smile corrected', calibrated=True)

    # Destriping
    water_mask = cal_func.make_mask(cube, sat_val_scale=0.9)
    destriping_correction_matrix_calibrated = cal_func.get_destriping_correction_matrix(cube_calibrated_smile_corrected, water_mask, plot)
    # if plot:
    #     show_bip.show_bip_cube(destriping_correction_matrix_calibrated, 500, 510, image_height, num_frames, 'Destriped', calibrated=True)

    # Save radiometric calibration matrix
    np.savetxt('Coefficients/destriping_correction_matrix_HYPSO-1_684spatial_1080bands.csv', destriping_correction_matrix_calibrated, delimiter=",") 

    return None



if __name__ == '__main__':
    main()