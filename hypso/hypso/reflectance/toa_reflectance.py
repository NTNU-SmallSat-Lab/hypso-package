
from importlib.resources import files
from dateutil import parser
import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, vstack, csr_matrix



def compute_toa_reflectance(sensor_wavelengths,
                            sensor_fwhm,
                            bin_factor: int,
                            toa_radiance: np.ndarray,
                            iso_time,
                            solar_zenith_angles,
                            use_thuillier = False
                            ) -> xr.DataArray:


    if use_thuillier:
        ssi, solar_wavelengths = load_thuillier_ssi()
    else:
        ssi, solar_wavelengths= load_ssi()






    srfs_csr, truncated_ssi, truncated_solar_wavelengths  = compute_srf(ssi=ssi,
                                                        solar_wavelengths=solar_wavelengths,
                                                        sensor_wavelengths=sensor_wavelengths,
                                                        sensor_fwhm=sensor_fwhm,
                                                        )
    

    binned_sensor_wavelengths = bin_sensor_wavelengths(sensor_wavelengths, bin_factor)

    binned_srfs_csr = bin_srf(srfs_csr, bin_factor)


    esun_list = compute_esun(srfs_csr=binned_srfs_csr, ssi=truncated_ssi, method="sparse")
    #esun_list = compute_esun(srfs_csr=binned_srfs_csr, ssi=truncated_ssi, method="vectorized")
    #esun_list = compute_esun(srfs_csr=binned_srfs_csr, ssi=truncated_ssi, method="loop")


    scene_date = parser.isoparse(iso_time)
    julian_day = scene_date.timetuple().tm_yday


    toa_reflectance = np.empty_like(toa_radiance)

    for band, esun in enumerate(esun_list):

        # Earth-Sun distance scaler (from day of year) using julian date
        # (R/R_0) earth-sun distance divided by average earth-sun distance
        # http://physics.stackexchange.com/questions/177949/earth-sun-distance-on-a-given-day-of-the-year
        # 4 is when earth reaches perihelion, day 4 for 2025
        sun_distance_scaler = 1 - 0.01672 * np.cos(0.9856 * (julian_day - 4))  

        # Get toa_reflectance
        # equation for "Normalized reflectances" found here:
        # https://oceanopticsbook.info/view/atmospheric-correction/normalized-reflectances 
        solar_angle_correction = np.cos(np.radians(solar_zenith_angles))
        multiplier = (esun * solar_angle_correction) / (np.pi * sun_distance_scaler ** 2)
        
        toa_reflectance[:, :, band] = toa_radiance[:, :, band] / multiplier







    if True:
        import csv
        import matplotlib.pyplot as plt

        ssi_values = np.array(ssi)

        with open('ssi_data_tsis.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Wavelength', 'SSI'])  # Header
            for wl, ssi_value in zip(solar_wavelengths, ssi_values):
                writer.writerow([wl, ssi_value])

        with open('esun_data_tsis.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Wavelength', 'ESUN'])  # Header
            for wl, esun_value in zip(binned_sensor_wavelengths, esun_list):
                writer.writerow([wl, esun_value])


        plt.plot(solar_wavelengths, ssi, label='TSIS-1 SSI', linewidth=0.15)
        plt.plot(np.array(binned_sensor_wavelengths), np.array(esun_list), label='HYPSO-2 $F_0$')

        plt.xlim(350, 850)
        plt.ylim(0,2500)
        plt.legend(loc='upper right')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Solar Spectral Irradiance [W m$^{-2}$ nm$^{-1}$]')
        plt.tight_layout()
        plt.savefig('1_hypso_spectrum_tsis.png')

        plt.close



    return toa_reflectance, srfs_csr, truncated_ssi, truncated_solar_wavelengths, esun_list







def compute_srf(ssi,
                solar_wavelengths,
                sensor_wavelengths,
                sensor_fwhm,
                ):


    fwhm_nm = sensor_fwhm

    # Calculate sigma from FWHM
    sigma_nm = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))


    # Find indices of sensor bands in solar wavelength array
    sensor_band_indices = [np.abs(solar_wavelengths - w).argmin() for w in sensor_wavelengths]


    # Determine truncation limits for the entire SSI based on first and last bands
    first_band_center = solar_wavelengths[sensor_band_indices[0]]
    last_band_center = solar_wavelengths[sensor_band_indices[-1]]


    # Calculate 3 sigma limits for first and last bands
    first_band_start = first_band_center - (3 * sigma_nm[0])
    last_band_end = last_band_center + (3 * sigma_nm[-1])
    
    # Find indices for truncation
    start_ssi_idx = np.abs(solar_wavelengths - first_band_start).argmin()
    end_ssi_idx = np.abs(solar_wavelengths - last_band_end).argmin()

    # Truncate SSI and corresponding wavelengths
    truncated_ssi = ssi[start_ssi_idx:end_ssi_idx + 1]
    truncated_solar_wavelengths = solar_wavelengths[start_ssi_idx:end_ssi_idx + 1]

    # Adjust sensor band indices to work with truncated array
    # Subtract start_ssi_idx to get positions in truncated array
    adjusted_band_indices = [idx - start_ssi_idx for idx in sensor_band_indices]

    n_bands = len(sensor_wavelengths)
    n_truncated_solar_wavelengths = len(truncated_solar_wavelengths)


    # Initialize sparse matrix in LIL format for efficient construction
    aligned_srfs_sparse = lil_matrix((n_bands, n_truncated_solar_wavelengths), dtype=np.float32)


    for i, (adjusted_idx, center_wavelength) in enumerate(zip(adjusted_band_indices, sensor_wavelengths)):

        # Calculate 3 sigma range for this band
        start_lambda = center_wavelength - (3 * sigma_nm[i])
        end_lambda = center_wavelength + (3 * sigma_nm[i])

        # Find indices directly in the truncated wavelength array (only searching within truncated range)
        start_idx = np.abs(truncated_solar_wavelengths - start_lambda).argmin()
        end_idx = np.abs(truncated_solar_wavelengths - end_lambda).argmin()

        # Get the wavelengths for this SRF from truncated array
        srf_wavelengths = truncated_solar_wavelengths[start_idx:end_idx + 1]

        # Create x-values for Gaussian (centered at 0)
        gx = np.linspace(-3 * sigma_nm[i], 3 * sigma_nm[i], len(srf_wavelengths))

        # Create Gaussian SRF (peak = 1.0)
        gaussian_srf = np.exp(-(gx / sigma_nm[i]) ** 2 / 2)

        # Store in aligned arrays (both dense and sparse)
        aligned_srfs_sparse[i, start_idx:end_idx + 1] = gaussian_srf

    # Convert to CSR for efficient row access
    srfs_csr = aligned_srfs_sparse.tocsr()


    return srfs_csr, truncated_ssi, truncated_solar_wavelengths 


def bin_sensor_wavelengths(sensor_wavelengths, bin_factor):
    return sensor_wavelengths.reshape(-1, bin_factor).mean(axis=1).reshape(-1)


def bin_srf(srfs_csr, bin_factor, truncated_solar_wavelengths=None):
    """
    Bin neighboring SRFs by adding them element-wise.
    
    Parameters:
    -----------
    srfs_csr : scipy.sparse.csr_matrix
        Sparse CSR matrix of SRFs (n_bands times n_wavelengths)
    bin_factor : int
        Number of neighboring bands to bin together
    truncated_solar_wavelengths : array, optional
        Wavelength array for plotting/debugging

    Returns:
    --------
    binned_srfs : scipy.sparse.csr_matrix
        Binned SRFs (n_bands//bin_factor times n_wavelengths)
    bin_indices : list
        List of tuples containing the original band indices in each bin
    """
    
    n_bands, n_wavelengths = srfs_csr.shape
    
    # Calculate number of bins
    n_bins = n_bands // bin_factor
    if n_bands % bin_factor != 0:
        print(f"[WARNING] {n_bands} bands not evenly divisible by bin_factor {bin_factor}")
        print(f"[WARNING] Truncating to {n_bins * bin_factor} bands")
        # Truncate to make evenly divisible
        srfs_csr = srfs_csr[:n_bins * bin_factor, :]
        n_bands = n_bins * bin_factor
    
    print(f"[INFO] Binning {n_bands} bands into {n_bins} bins (factor={bin_factor})")
    
    # Initialize list to store binned SRFs
    binned_rows = []
    bin_indices = []
    
    for bin_idx in range(n_bins):
        start_band = bin_idx * bin_factor
        end_band = start_band + bin_factor
        
        # Get the slice of bands for this bin
        band_slice = srfs_csr[start_band:end_band, :]
        
        # Sum the bands element-wise
        # .sum(axis=0) returns a matrix, .A1 flattens to 1D array
        binned_srf = band_slice.sum(axis=0).A1
        
        # Store as sparse row
        binned_rows.append(csr_matrix(binned_srf))
        
        # Keep track of which original bands are in this bin
        bin_indices.append(list(range(start_band, end_band)))
        
        if (bin_idx + 1) % 10 == 0:
            print(f"Processed {bin_idx + 1}/{n_bins} bins")
    
    # Stack all binned rows into a single CSR matrix
    binned_srfs_csr = vstack(binned_rows, format='csr')
    
    print(f"[INFO] Binned SRF matrix shape: {binned_srfs_csr.shape}")
    #print(f"Non-zero elements: {binned_srfs_csr.nnz}")
    #print(f"Average non-zero per binned band: {binned_srfs_csr.nnz / n_bins:.1f}")
    

    # Optional visualization
    if truncated_solar_wavelengths is not None:
        visualize_srf_binning(srfs_csr, binned_srfs_csr, bin_factor, 
                         truncated_solar_wavelengths, bin_indices)
    
    return binned_srfs_csr


def visualize_srf_binning(original_srfs, binned_srfs, bin_factor, 
                     wavelengths, bin_indices, n_examples=3):
    """
    Visualize the binning process with examples.
    """
    fig, axes = plt.subplots(n_examples, 2, figsize=(14, 4*n_examples))
    
    # Pick some example bins to visualize
    example_bins = np.linspace(0, len(bin_indices)-1, n_examples, dtype=int)
    
    for idx, bin_num in enumerate(example_bins):
        # Plot original bands in this bin
        ax = axes[idx, 0]
        start, end = bin_indices[bin_num][0], bin_indices[bin_num][-1]
        
        for band_idx in range(start, end + 1):
            band_srf = original_srfs[band_idx, :].toarray().flatten()
            ax.plot(wavelengths, band_srf, alpha=0.5, 
                   label=f'Band {band_idx}' if idx == 0 else None)
        
        ax.set_title(f'Original Bands {start}-{end}')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('SRF')
        if idx == 0:
            ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.3)
        
        # Plot binned result
        ax = axes[idx, 1]
        binned_srf = binned_srfs[bin_num, :].toarray().flatten()
        ax.plot(wavelengths, binned_srf, 'r-', linewidth=2, 
               label=f'Binned Band {bin_num}')
        
        # Overlay individual bands faintly for comparison
        for band_idx in range(start, end + 1):
            band_srf = original_srfs[band_idx, :].toarray().flatten()
            ax.plot(wavelengths, band_srf, 'k-', alpha=0.2)
        
        ax.set_title(f'Binned Band {bin_num} (sum of {bin_factor} bands)')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('SRF')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def compute_esun_from_csr(srfs_csr, ssi, band_indices=None):
    """
    Compute Esun values from CSR-formatted SRFs and truncated SSI.
    
    Parameters:
    -----------
    srfs_csr : scipy.sparse.csr_matrix
        CSR matrix of aligned SRFs (n_bands × n_wavelengths)
    truncated_ssi : numpy array
        Truncated solar spectral irradiance (aligned with SRF wavelengths)
    band_indices : list or array, optional
        Specific band indices to compute. If None, compute for all bands.
    
    Returns:
    --------
    esun_list : list
        Esun values for each band
    """
    
    if band_indices is None:
        band_indices = range(srfs_csr.shape[0])
    
    esun_list = []
    
    for i in band_indices:
        # Get the SRF for this band (as dense array)
        # .A1 flattens the matrix to 1D array
        gaussian_srf = srfs_csr[i, :].toarray().flatten()
        
        # Find where SRF is non-zero (for numerical stability)
        nonzero_mask = gaussian_srf > 0
        
        if np.any(nonzero_mask):
            # Extract non-zero portion
            srf_nonzero = gaussian_srf[nonzero_mask]
            ssi_nonzero = ssi[nonzero_mask]
            
            # Calculate sum of SRF for normalization
            gaussian_srf_sum = np.sum(srf_nonzero)
            
            # Calculate weights
            srf_weights = srf_nonzero / gaussian_srf_sum
            
            # Calculate Esun value
            esun_value = np.sum(ssi_nonzero * srf_weights)
        else:
            # Handle case where SRF is all zeros (shouldn't happen)
            esun_value = 0.0
            print(f"Warning: Band {i} has all-zero SRF")
        
        esun_list.append(esun_value)
    
    return esun_list


# More efficient vectorized version
def compute_esun_vectorized(srfs_csr, ssi):
    """
    Vectorized computation of Esun values (faster for many bands).
    
    Parameters:
    -----------
    srfs_csr : scipy.sparse.csr_matrix
        CSR matrix of aligned SRFs (n_bands × n_wavelengths)
    ssi : numpy array
        Truncated solar spectral irradiance
    
    Returns:
    --------
    esun_list : numpy array
        Esun values for all bands
    """
    
    # Get dimensions
    n_bands, n_wavelengths = srfs_csr.shape
    
    # Pre-allocate result array
    esun_values = np.zeros(n_bands)
    
    # Process each band (still need loop due to variable non-zero patterns)
    for i in range(n_bands):
        # Get SRF row as dense array
        srf_row = srfs_csr[i, :].toarray().flatten()
        
        # Find non-zero elements
        nonzero_idx = srf_row > 0
        
        if np.any(nonzero_idx):
            # Extract non-zero portions
            srf_nonzero = srf_row[nonzero_idx]
            ssi_nonzero = ssi[nonzero_idx]
            
            # Normalize and compute
            srf_sum = np.sum(srf_nonzero)
            esun_values[i] = np.sum(ssi_nonzero * (srf_nonzero / srf_sum))
    
    return esun_values.tolist()


# Ultra-efficient version using sparse matrix operations
def compute_esun_sparse_efficient(srfs_csr, ssi):
    """
    Most efficient version using sparse matrix operations.
    This avoids converting to dense arrays entirely.
    
    Parameters:
    -----------
    srfs_csr : scipy.sparse.csr_matrix
        CSR matrix of aligned SRFs
    ssi : numpy array
        Truncated solar spectral irradiance
    
    Returns:
    --------
    esun_list : numpy array
        Esun values for all bands
    """
    
    n_bands = srfs_csr.shape[0]
    esun_values = np.zeros(n_bands)
    
    # CSR format stores data in row-major order with indices
    # We can iterate through rows efficiently
    for i in range(n_bands):
        # Get the start and end indices for row i in the data arrays
        start_idx = srfs_csr.indptr[i]
        end_idx = srfs_csr.indptr[i + 1]
        
        if start_idx < end_idx:  # Row has non-zero elements
            # Get the column indices and values for this row
            col_indices = srfs_csr.indices[start_idx:end_idx]
            srf_values = srfs_csr.data[start_idx:end_idx]
            
            # Get corresponding SSI values
            ssi_values = ssi[col_indices]
            
            # Calculate sum of SRF for normalization
            srf_sum = np.sum(srf_values)
            
            # Calculate Esun
            esun_values[i] = np.sum(ssi_values * (srf_values / srf_sum))
    
    return esun_values.tolist()


def compute_esun(srfs_csr, ssi, method='sparse'):
    """
    Wrapper function to compute Esun with different methods.
    
    Parameters:
    -----------
    srfs_csr : scipy.sparse.csr_matrix
        CSR matrix of aligned SRFs
    truncated_ssi : numpy array
        Truncated solar spectral irradiance
    method : str
        'sparse' - most memory efficient (recommended)
        'vectorized' - good balance
        'loop' - original loop style
    
    Returns:
    --------
    esun_list : list
        Esun values for all bands
    """
    
    if method == 'sparse':
        esun_list = compute_esun_sparse_efficient(srfs_csr, ssi)
    elif method == 'vectorized':
        esun_list = compute_esun_vectorized(srfs_csr, ssi)
    else:  # 'loop'
        esun_list = compute_esun_from_csr(srfs_csr, ssi)
    
    return esun_list








def load_ssi():

    # Load the NetCDF file
    #solar_data_path = str(files('hypso.reflectance').joinpath("hybrid_reference_spectrum_p1nm_resolution_c2022-11-30_with_unc.nc"))
    #ds = xr.open_dataset(solar_data_path)
    #solar_x = ds["Vacuum Wavelength"].values
    #solar_y = ds["SSI"].values * 1000 # convert to milliwatts
    #ds.close()

    # Load .npz file containing the pre-processed solar spectrum irradiance (SSI). It is generating using the 'write_ssi_npz.py' script. 
    # The SSI is truncated to the visible spectrum range covering the HYPSO-1 & -2 bands
    # The SSI is from the TSIS-1 SSI v2 file 'hybrid_reference_spectrum_p005nm_resolution_c2022-11-30_with_unc.nc'. A copy is in the Git repository.
    # Source & download: https://lasp.colorado.edu/lisird/data/tsis1_hsrs_p1nm
    solar_data_path = str(files('hypso.reflectance').joinpath("hybrid_reference_spectrum_p005nm_resolution_c2022-11-30_with_unc.npz"))
    ds = np.load(solar_data_path)

    solar_wavelengths = ds["solar_x"]
    ssi = ds["solar_y"] * 1000 # convert to milliwatts


    return ssi, solar_wavelengths




def load_thuillier_ssi():

    print("[WARNING] Using Thuillier SSI reference spectrum for ToA reflectance processing!")
    solar_data_path = str(files('hypso.reflectance').joinpath("Solar_irradiance_Thuillier_2002.csv"))
    solar_df = pd.read_csv(solar_data_path)
    ssi = np.array(solar_df['mW/m2/nm'])
    solar_wavelengths = np.array(solar_df['nm'])

    return ssi, solar_wavelengths