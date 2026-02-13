import numpy as np


def run_srem_atmospheric_correction(satobj, VERBOSE=True):

    if VERBOSE: 
        print("[INFO] Running SREM atmospheric correction")



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

    relative_azimuth_angles = satobj.relative_azimuth_angles

    iso_time = satobj.iso_time

    wavelengths = satobj.wavelengths

    height, width, bands = l1d_cube.shape

    for band in range(0,bands):

        rho_TOA = l1d_cube[:,:,band]

        lambda_wl = wavelengths[band] / 1000 # wavelength in micrometers


        # Step I: Compute scattering angle

        # Convert to radians
        sz_rad = np.radians(solar_zenith_angles)
        vz_rad = np.radians(sat_zenith_angles)
        raz_rad = np.radians(relative_azimuth_angles)
        
        # SREM formula: cos(Θ) = -cos(θs)*cos(θv) - sin(θs)*sin(θv)*cos(Δφ)
        cos_theta = (-np.cos(sz_rad) * np.cos(vz_rad) - np.sin(sz_rad) * np.sin(vz_rad) * np.cos(raz_rad))
        
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        theta_sc = np.arccos(cos_theta)


        # Step II: Compute Rayleigh phase function using Equation (10)
        A = 0.9587256
        B = 1 - A
        P_R = ((3*A)/(4+B)) * (1 + np.cos(theta_sc)**2)

    
        # Step III: Compute atmospheric reflectance due to Rayleigh scattering rho_R

        # Compute air mass M using Equation (8)
        mu_s = np.cos(sz_rad) # cosine of solar zenith angle, mu_s
        mu_v = np.cos(vz_rad) # cosine of the sensor zenith angle, mu_v
        M = (1/mu_s) + (1/mu_v) # Air mass, M

        # Compute Rayleigh optical depth using Equation (9)
        tau_r_coeff1 = 0.008569
        tau_r_coeff2 = 0.0113
        tau_r_coeff3 = 0.0113
        tau_r = tau_r_coeff1 * (lambda_wl**-4) * (1 + tau_r_coeff2 * (lambda_wl**-2) + tau_r_coeff3 * (lambda_wl**-4))

        # Compute rho_R using Equation (7)
        rho_R = P_R * ((1 - np.exp(-M*tau_r)) / (4*(mu_s + mu_v)))


        # Step IV: Compute atmospheric backscattering ratio S_atm

        # Compute S_atm using Equation (11)
        S_atm = 0.92 * tau_r * np.exp(-tau_r)

        # Compute atmospheric transmittance of sun-surface path (downward) using Equation (12)
        T_s_coeff1 = np.exp(-tau_r/mu_s)
        T_s_coeff2 = np.exp(0.52*tau_r/mu_s)
        T_s = T_s_coeff1 + T_s_coeff1 * (T_s_coeff2 - 1)

        # Compute atmospheric transmittance of surface-sensor path (upward) using Equation (13)
        T_v_coeff1 = np.exp(-tau_r/mu_s)
        T_v_coeff2 = np.exp(0.52*tau_r/mu_s)
        T_v = T_v_coeff1 + T_v_coeff1 * (T_v_coeff2 - 1)


        numerator = rho_TOA - rho_R
        denominator = (rho_TOA - rho_R)*S_atm + T_s*T_v

        rho_s = numerator/denominator
        
        cube[:,:,band] = rho_s

    return cube
