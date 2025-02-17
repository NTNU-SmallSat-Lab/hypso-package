'''Functions that can be used in the HYPSO calibration pipeline.

2023-03-20 MBH'''

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import skimage.morphology as morph
import copy

# Local files
import sys
sys.path.insert(0, '../Tools')
import utilities as util


def crop_and_bin_matrix(matrix, x_start, x_stop, y_start, y_stop, bin_x=1, bin_y=1):
    ''' Crops matrix to AOI. Bins matrix so that the average value in the bin_x 
    number of pixels is stored.
    '''
    # Crop to selected AOI
    new_matrix = matrix[y_start:y_stop, x_start:x_stop]
    height, width = new_matrix.shape

    # If bin is set to 0 or negative we assume this means no binning, aka bin=1
    if bin_x < 1:
        bin_x = 1
    if bin_y < 1:
        bin_y = 1

    # Bin spectral direction
    if bin_x != 1:
        width_binned = int(width/bin_x)
        matrix_cropped_and_binned = np.zeros((height,width_binned))
        for i in range(width_binned):
            this_pixel_sum = 0
            for j in range(bin_x):
                this_pixel_value = new_matrix[:,i*bin_x+j]
                this_pixel_sum += this_pixel_value
            average_pixel_value = this_pixel_sum/bin_x
            matrix_cropped_and_binned[:,i] = average_pixel_value
        new_matrix = matrix_cropped_and_binned

    # Bin spatial direction
    if bin_y != 1:
        height_binned = int(height/bin_y)
        matrix_binned_spatial = np.zeros((height_binned,width_binned))
        for i in range(height_binned):
            this_pixel_sum = 0
            for j in range(bin_y):
                this_pixel_value = new_matrix[i*bin_y+j,:]
                this_pixel_sum += this_pixel_value
            average_pixel_value = this_pixel_sum/bin_y
            matrix_binned_spatial[i,:] = average_pixel_value/bin_y
        new_matrix = matrix_binned_spatial

    return new_matrix


def make_overexposed_mask(cube, over_exposed_lim=4094):
    ''' Makes mask for spatial image, so that all good values (not masked) are 
    not overexposed for all wavelengths. 
    
    1 in mask = good pixel
    0 in mask = bad pixel (overexposed)
    
    To apply the mask, just multiply each spatial frame with the mask.
    '''
    num_frames, image_height, image_width = cube.shape
    mask = np.ones([num_frames, image_height])
    for i in range(image_width):
        this_spatial_im = cube[:,:,i]
        mask = np.where(np.array(this_spatial_im) > over_exposed_lim, 0, mask)

    return mask


def make_mask(cube, sat_val_scale=0.25, plot=False):
    ''' Mask values based on all values in cube. Used with destriping.

    For water mask: sat_val_scale=0.25
    For overexposed mask: sat_val_scale=0.9 
    '''
    cube_sum = np.sum(cube, axis=2)#/num_frames
    sat_value = cube_sum.max()*sat_val_scale
    mask = cube_sum > sat_value
    if plot:
        fig, ax = plt.subplots()
        plt.imshow(cube_sum)
        plt.colorbar()
        plt.title('Cube summed (to look at values)')
        fig, ax = plt.subplots()
        plt.imshow(cube_sum > sat_value)
        plt.colorbar()
        plt.title('Mask')
    return mask


def apply_radiometric_calibration(frame, exp, background_value, radiometric_calibration_coefficients):
    ''' Assumes input is 12-bit values, and that the radiometric calibration
    coefficients are the same size as the input image.
    
    Note: radiometric calibration coefficients have original size (684,1080),
    matching the "normal" AOI of the HYPSO-1 data (with no binning).
    '''
    frame = frame - background_value
    frame_calibrated = frame * radiometric_calibration_coefficients / exp
    
    return frame_calibrated


def calibrate_cube(cube, exp, radiometric_calibration_matrix, background_value):
    ''' Calibrate cube.

    Only includes radiometric calibration.

    Assumes the coefficients has been adjusted to the frame size (cropped and
    binned), and that the data cube contains 12-bit values.
    ''' 
    image_height, image_width = radiometric_calibration_matrix.shape

    ## Radiometric calibration
    num_frames = cube.shape[0]
    cube_calibrated = np.zeros([num_frames, image_height, image_width])
    for i in range(num_frames):
        frame = cube[i,:,:]
        frame_calibrated = apply_radiometric_calibration(frame, exp, background_value, radiometric_calibration_matrix)
        cube_calibrated[i,:,:] = frame_calibrated
    return cube_calibrated


def calibrate_and_correct_cube(cube, exp, radiometric_calibration_matrix, background_value, spectral_band_matrix, destriping_matrix):
    ''' Calibrate cube.

    Includes:
    - Radiometric calibration
    - Smile correction
    - Destriping

    Assumes all coefficients has been adjusted to the frame size (cropped and
    binned), and that the data cube contains 12-bit values.
    '''

    ## Radiometric calibration
    cube_calibrated = calibrate_cube(cube, exp, radiometric_calibration_matrix, background_value)

    ## Smile correction
    cube_smile_corrected = smile_correction_cube(cube_calibrated, spectral_band_matrix)

    ## Destriping
    cube_destriped = apply_destriping_correction_matrix(cube_smile_corrected, destriping_matrix)
    
    return cube_destriped


def smile_correction_one_row(row, w, w_ref):
    ''' Use cubic spline interpolation to resample one row onto the correct
    wavelengths/bands from a reference wavelength/band array to correct for
    the smile effect.
    '''
    row_interpolated = interpolate.splrep(w, row)
    row_corrected = interpolate.splev(w_ref, row_interpolated)
    # Set values for wavelengths below 400 nm to zero
    for i in range(len(w_ref)):
        w = w_ref[i]
        if w < 400:
            row_corrected[i] = 0
        else:
            break
    return row_corrected


def smile_correction_one_frame(frame, spectral_band_matrix):
    ''' Run smile correction on each row in a frame, using the center row as 
    the reference wavelength/band for smile correction.
    '''
    image_height, image_width = frame.shape
    center_row_no = int(image_height/2)
    w_ref = spectral_band_matrix[center_row_no]
    frame_smile_corrected = np.zeros([image_height, image_width])
    for i in range(image_height): # For each row
        this_w = spectral_band_matrix[i]
        this_row = frame[i]
        # Correct row
        row_corrected = smile_correction_one_row(this_row, this_w, w_ref)
        frame_smile_corrected[i,:] = row_corrected
    return frame_smile_corrected


def smile_correction_cube(cube, spectral_band_matrix):
    ''' Run smile correction on each frame in a cube, using the center row in 
    the frame as the reference wavelength/band for smile correction.
    '''
    num_frames, image_height, image_width = cube.shape
    cube_smile_corrected = np.zeros([num_frames, image_height, image_width])
    for i in range(num_frames):
        this_frame = cube[i,:,:]
        frame_smile_corrected = smile_correction_one_frame(this_frame, spectral_band_matrix)
        cube_smile_corrected[i,:,:] = frame_smile_corrected
    return cube_smile_corrected


def get_destriping_correction_matrix(cube, water_mask, plot=False, plot_min_val=-1, plot_max_val=1):
    ''' Use masked ocean cube to create a destriping correction matrix 
    (cumulative correction frame). Must be calibrated with the same calibration
    coefficients (and smile correction or other correction steps) that will be
    used before the destriping correction is used on any other cube.

    Based on Joe's jupyter code.
    '''
    num_frames, image_height, image_width = cube.shape
    # Determine correction
    wm = morph.dilation(water_mask, morph.square(7))
    diff = cube[:,1:,100] - cube[:,:-1,100] 
    wm_t = wm[:,:-1]
    wm_t[wm[:,1:]] = 1 
    diff[wm_t]=0
    diff = np.zeros((num_frames,image_height-1,image_width))
    diff[:] = cube[:,1:,:] - cube[:,:-1,:]
    diff[wm_t]=0
    corrections = np.zeros((image_height,image_width))
    for i in range(0,image_height-1):
        corrections[i,:] = np.median(diff[:,i][~wm_t[:,i]], axis=0)
    corrections[:] -= np.mean(corrections, axis=0)
    cumulative = np.zeros((image_height, image_width))
    cumulative[0] = corrections[0]
    for i in range(1,image_height):
        cumulative[i] = corrections[i] + cumulative[i-1]
    if plot:
        fig, ax = plt.subplots()
        plt.imshow(cumulative, vmin=plot_min_val, vmax=plot_max_val)
        plt.xlabel('Spectral axis [pixel]')
        plt.ylabel('Spatial axis [pixel]')
        plt.title('Cumulative correction frame')
        plt.colorbar()
    return cumulative


def apply_destriping_correction_matrix(cube, destriping_correction_matrix):
    ''' Apply destriping correction matrix. '''
    cube_delined = copy.deepcopy(cube)
    cube_delined[:,1:] -= destriping_correction_matrix[:-1]
    return cube_delined

