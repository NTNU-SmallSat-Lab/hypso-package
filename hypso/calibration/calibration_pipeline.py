'''Suggestion for calibration pipeline for HYPSO data.

2023-03-07 MBH'''

import numpy as np
import matplotlib.pyplot as plt

# Local files
import sys
sys.path.insert(0, '../Tools')
import utilities as util
import read_images
import calibration_pipeline_functions as cal_func
import show_bip

def main():
    
    #############
    ##  INPUT  ##
    #############

    # ## FINNMARK
    # # Want to use this one bc Joe uses it for destriping, but the max value is weird (44682/9=4964)
    # # Path and filename of raw cube
    # path = '../../../Data/HYPSO-1/finnmark/'
    # cube_filename = path + '2022-08-06_T103522_grieg_finnmark_2/grieg_finnmark_2022_08_06T10_35_22.bip' 

    # # Metadata from capture_config.ini
    # flags = 0x00000200
    # camera_ID = 1
    # frame_count = 956
    # exposure = 30.0063
    # fps = 22
    # row_count = 684
    # column_count = 1080
    # sample_divisor = 1
    # bin_factor = 9
    # aoi_x = 428
    # aoi_y = 266
    # gain = 0
    # temp_log_period_ms = 10000

    ## FROHAVET
    # Want to use this instead of Finnmark cube
    # Path and filename of raw cube
    path = '../../../Data/HYPSO-1/frohavet/'
    cube_filename = path + '2022-05-17/frohavet-2022-05-17.bip' 

    # Metadata from capture_config.ini
    flags = 0x00000200
    camera_ID = 1
    frame_count = 956
    exposure = 49.7349
    fps = 20
    row_count = 684
    column_count = 1080
    sample_divisor = 1
    bin_factor = 9
    aoi_x = 428
    aoi_y = 266
    gain = 0
    temp_log_period_ms = 10000

    # ## SVALBARD
    # # This one is a good example of cropped and binned data, with some overexposed values
    # # Path and filename of raw cube
    # path = '../../../Data/HYPSO-1/svalbard/'
    # cube_filename = path + '2022-07-18/svalbard.bip' 

    # # Metadata from capture_config.ini
    # flags = 0x00000200
    # camera_ID = 1
    # frame_count = 956
    # exposure = 30.0063
    # fps = 22
    # row_count = 684
    # column_count = 1080
    # sample_divisor = 1
    # bin_factor = 9
    # aoi_x = 428
    # aoi_y = 266
    # gain = 0
    # temp_log_period_ms = 10000

    # ## SAHARA FULL FRAME
    # # Full frame of uniform area, but low values (no overexposed)
    # # Path and filename of raw cube
    # path = '../../../Data/HYPSO-1/sahara/'
    # cube_filename = path + 'sahara1/sahara1_wo_binning_e10ms.bip' 

    # # Metadata from capture_config.ini
    # flags = 0x00000001
    # camera_ID = 1
    # frame_count = 2
    # exposure = 10.0029
    # fps = 1
    # row_count = 1216
    # column_count = 1936
    # sample_divisor = 1
    # bin_factor = 1
    # aoi_x = 0
    # aoi_y = 0
    # gain = 0
    # temp_log_period_ms = 10000

    # ## MOON FULL FRAME
    # # Full frame, with overexposed (moon) and lots of darkness (space)
    # # Path and filename of raw cube
    # path = '../../../Data/HYPSO-1/moon/'
    # cube_filename = path + 'full-frame/20221011_CaptureDL_moon_22_23_18/moonbuffer.bip' 

    # # Metadata from capture_config.ini
    # flags = 0x00000201
    # camera_ID = 2
    # frame_count = 33
    # exposure = 100.006
    # fps = 4
    # row_count = 1216
    # column_count = 1936
    # sample_divisor = 1
    # bin_factor = 1
    # aoi_x = 0
    # aoi_y = 0
    # gain = 0
    # temp_log_period_ms = 10000

    # Metadata
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

    # Path and filename of calibration files
    coeff_path = 'Coefficients/'
    spectral_band_matrix = util.readCsvFile(coeff_path+'spectral_calibration_matrix_HYPSO-1.csv')
    radiometric_calibration_matrix = util.readCsvFile(coeff_path+'radiometric_calibration_matrix_HYPSO-1.csv')
    background_value = 8 # TODO: can improve this one with adjusted values for exp/temp
    if image_height == 684 and image_width == 120:
        do_destriping = True
        destriping_correction_matrix = util.readCsvFile(coeff_path+'destriping_correction_matrix_HYPSO-1_684spatial_120bands.csv') # NOTE: this is for cropped to AOI and bin = 9
    elif image_height == 684 and image_width == 1080:
        do_destriping = True
        destriping_correction_matrix = util.readCsvFile(coeff_path+'destriping_correction_matrix_HYPSO-1_684spatial_1080bands.csv')
    else:
        do_destriping = False
        print("We don't have the destriping correction coefficients for this frame configuration yet.")
        print("Will not do destriping.")
        
    ################

    # Read cube
    cube = read_images.read_bip_cube(cube_filename, image_width, image_height)
    cube = cube[:,:,::-1] # Flip HYPSO-1 cube
    show_bip.show_bip_cube(cube, 50, 51, image_height, num_frames, 'Original cube')

    # Adjust calibration matrices to cropped and binned frames
    print('Original max value of the cube is: %.2f'%np.max(cube))
    # Scale to get 12-bit values
    # The values are summed when saving binned cubes, so divide by binned 
    # values should provide the original 12-bit counts
    cube = cube/bin_x

    # Crop and bin spectral calibration matrix
    spectral_band_matrix = cal_func.crop_and_bin_matrix(spectral_band_matrix, x_start, x_stop, y_start, y_stop, bin_x)

    # Crop and bin radiometric calibration matrix
    # Assumes that the cube is scaled correctly to 12-bit values 
    radiometric_calibration_matrix = cal_func.crop_and_bin_matrix(radiometric_calibration_matrix, x_start, x_stop, y_start, y_stop, bin_x)
    print('Max value of scaled cube is: %.2f'%np.max(cube))
    print('(The max value should be 4095 when overexposed.)')    

    # Look at wavelengths in middle row (no smile)
    center_row_no = int(image_height/2)
    center_line_w = spectral_band_matrix[center_row_no]
    print('Bands/center wavelengths in middle row: ', center_line_w)

    # Step 0: Overexposed mask
    # Mark all overexposed values in the spatial-spatial image as 0, all "good" values as 1, and save as a mask
    # It's easier to mask out overexposed values before the calibration
    # TODO: could also save mask as a cube, only masking the overexposed pixels in each frame
    overexp_mask = cal_func.make_overexposed_mask(cube, over_exposed_lim=4094)
    overexp_mask_filename = cube_filename[:-4] + '_overexp_mask.csv'
    np.savetxt(overexp_mask_filename, overexp_mask, delimiter=",") 
    # Plot overexposed mask
    fig, ax = plt.subplots()
    plt.imshow(overexp_mask, cmap='gray')
    plt.xlabel('Across-track [pixel]')
    plt.ylabel('Along-track [pixel]')
    # Set overexposed values to 0
    for i in range(image_width):
        cube[:,:,i] = cube[:,:,i]*overexp_mask
    show_bip.show_bip_cube(cube, 50, 51, image_height, num_frames, 'Overexposed values set to 0')

    # Step 1: Radiometric calibration
    cube_calibrated = cal_func.calibrate_cube(cube, exp, radiometric_calibration_matrix, background_value)
    show_bip.show_bip_cube(cube_calibrated, 50, 51, image_height, num_frames, 'Radiometric calibration applied', calibrated=True)

    # Step 2: Smile correction
    cube_smile_corrected = cal_func.smile_correction_cube(cube_calibrated, spectral_band_matrix)
    show_bip.show_bip_cube(cube_calibrated, 50, 51, image_height, num_frames, 'Smile correction applied', calibrated=True)

    # Step 3: Destriping
    if do_destriping:
        cube_destriped = cal_func.apply_destriping_correction_matrix(cube_smile_corrected, destriping_correction_matrix)
        show_bip.show_bip_cube(cube_destriped, 50, 51, image_height, num_frames, 'Destriped', calibrated=True)

    # # Step 1-3 in one function
    # cube_destriped = cal_func.calibrate_and_correct_cube(cube, exp, radiometric_calibration_matrix, background_value, spectral_band_matrix, destriping_correction_matrix)
    # show_bip.show_bip_cube(cube_destriped, 50, 51, image_height, num_frames, 'Destriped', calibrated=True)

    # Step 4: Reflectance

    # Make RGB composite

    # Save to file

    plt.show()

    return None



if __name__ == '__main__':
    main()