import numpy as np



def run_srem_oyam_atmospheric_correction(satobj, VERBOSE=True):

    try:
        # https://github.com/oyam/srem
        from srem import srem
    except Exception as ex:
        print("[ERROR] Unable to load SREM OYAM atmospheric correction. Please verify the SREM OYAM package is installed. Info: https://github.com/oyam/srem")
        print(ex)
        return None


    if VERBOSE: 
        print("[INFO] Running SREM Oyam atmospheric correction")


    wavelengths = satobj.wavelengths

    l1d_cube = satobj.l1d_cube.to_numpy()
    
    cube = np.empty_like(l1d_cube)

    try:
        latitudes = satobj.latitudes_indirect
        longitudes = satobj.longitudes_indirect
    except Exception as ex:
        print(ex)
        print("[WARNING] Defaulting to direct georeferencing.")
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes

    solar_azimuth_angles = satobj.solar_azimuth_angles
    solar_zenith_angles = satobj.solar_zenith_angles

    sat_azimuth_angles = satobj.sat_azimuth_angles
    sat_zenith_angles = satobj.sat_zenith_angles


    wavelengths = satobj.wavelengths

    height, width, bands = l1d_cube.shape

    for band in range(0,bands):

        rho_TOA = l1d_cube[:,:,band]

        lambda_wl = wavelengths[band] / 1000 # wavelength in micrometers

        rho_s = srem(
            rho_TOA, # np.ndarray with shape of (height, width)
            lambda_wl, # float in micrometer
            solar_azimuth_angles, # float or np.ndarray with shape of (height, width)
            solar_zenith_angles, # float or np.ndarray with shape of (height, width)
            sat_azimuth_angles, # float or np.ndarray with shape of (height, width)
            sat_zenith_angles # float or np.ndarray with shape of (height, width)
        )
        
        cube[:,:,band] = rho_s

    return cube
