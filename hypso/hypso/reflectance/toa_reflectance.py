
import warnings
from importlib.resources import files
from dateutil import parser
import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, vstack, csr_matrix
from tqdm import tqdm


def compute_toa_reflectance(sensor_wavelengths,
                            sensor_fwhm,
                            bin_factor: int,
                            toa_radiance: np.ndarray,
                            iso_time,
                            solar_zenith_angles,
                            use_thuillier: bool = False,
                            generate_figures: bool = False
                            ) -> xr.DataArray:
    """Legacy wrapper, superseded by hypso.reflectance.spectral_response (see
    that module's docstring for what changed and why). The SRF/esun block that
    used to live inline here is now compute_spectral_response(grid=
    "native-truncated"); the reflectance math is compute_reflectance below.
    Kept because callers still expect the original 7-tuple return - migrating
    them to consume a SpectralResponse directly happens in the later
    AC-connector pass (REFACTOR_PROGRESS.md).

    Returns (toa_reflectance, effective_fwhm, binned_srfs_csr, truncated_ssi,
    truncated_solar_wavelengths, esun, binned_sensor_wavelengths) - unchanged.
    """
    # Deferred import: spectral_response imports this module's low-level
    # helpers (compute_srf/bin_srf/...), so a module-level import here would
    # be circular.
    from .spectral_response import compute_spectral_response

    sr = compute_spectral_response(
        band_centers_unbinned=sensor_wavelengths,
        fwhm_unbinned=sensor_fwhm,
        bin_factor=bin_factor,
        ssi_source="thuillier" if use_thuillier else "tsis",
        grid="native-truncated",
        generate_figures=generate_figures,
    )

    toa_reflectance = compute_reflectance(toa_radiance=toa_radiance, sr=sr,
                                          iso_time=iso_time,
                                          solar_zenith_angles=solar_zenith_angles)

    if generate_figures:
        # Debug/figure output over the FULL (untruncated) SSI - reloaded here
        # since the SpectralResponse deliberately carries only the grid the
        # SRFs live on.
        import csv

        if use_thuillier:
            ssi, solar_wavelengths = load_thuillier_ssi()
            ssi_name = "thuillier"
        else:
            ssi, solar_wavelengths = load_ssi()
            ssi_name = "tsis"

        ssi_values = np.array(ssi)

        with open('ssi_data_' + ssi_name + '.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Wavelength', 'SSI'])  # Header
            for wl, ssi_value in zip(solar_wavelengths, ssi_values):
                writer.writerow([wl, ssi_value])

        with open('esun_data_' + ssi_name + '.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Wavelength', 'ESUN'])  # Header
            for wl, esun_value in zip(sr.band_centers, sr.esun):
                writer.writerow([wl, esun_value])

        plt.plot(solar_wavelengths, ssi, label='TSIS-1 SSI', linewidth=0.15)
        plt.plot(np.array(sr.band_centers), np.array(sr.esun), label='HYPSO-2 $F_0$')

        plt.xlim(350, 850)
        plt.ylim(0,2500)
        plt.legend(loc='upper right')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Solar Spectral Irradiance [W m$^{-2}$ nm$^{-1}$]')
        plt.tight_layout()
        plt.savefig('1_hypso_spectrum_' + ssi_name + '.png')

        plt.close()

    return (toa_reflectance, sr.effective_fwhm, sr.srf, sr.ssi, sr.grid_wl,
            sr.esun, sr.band_centers)


def compute_reflectance(toa_radiance: np.ndarray, sr, iso_time,
                        solar_zenith_angles) -> np.ndarray:
    """TOA radiance -> TOA reflectance using a SpectralResponse's per-band
    esun. Extracted verbatim from the old compute_toa_reflectance body - same
    Earth-Sun-distance and solar-zenith math."""
    scene_date = parser.isoparse(iso_time)
    julian_day = scene_date.timetuple().tm_yday

    toa_reflectance = np.empty_like(toa_radiance)

    for band, esun in enumerate(sr.esun):

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

    return toa_reflectance


def compute_csiro_srfs(self, generate_figures: bool = False) -> xr.DataArray:
    """DEPRECATED legacy wrapper (bound as a HypsoCapture method), superseded by
    hypso.reflectance.spectral_response's compute_spectral_response(grid=
    "uniform-1000") - the CSIRO-variant computation body moved there verbatim.

    Deprecated because the whole csiro path is computed but consumed by
    nothing: its only caller is hypso-processing-pipeline's
    stage2_ac/process_capture.py (which invokes it before the Polymer
    generate_srf_nc/ssi/esun calls - but those read the OTHER attribute
    family, self.srf/srf_ssi/esun), and while the csiro_* attributes are
    persisted into L1D files by write/metadata_srf_group_writer.py, no code
    anywhere reads them back for any computation. Removal is planned once the
    pipeline drops its call (the later AC-connector pass); until then this
    keeps working exactly as before, populating the csiro_* attribute family
    and self.spectral_response_csiro.
    """
    warnings.warn(
        "compute_csiro_srfs() is deprecated: its results (the csiro_* "
        "attributes, persisted into L1D metadata/srf) are not consumed by any "
        "known code - the Polymer SRF/SSI/ESUN files are generated from the "
        "spectral_response/native-truncated path instead. It will be removed "
        "once hypso-processing-pipeline drops its call. Use "
        "hypso.reflectance.compute_spectral_response(grid='uniform-1000') "
        "directly if the uniform-grid variant is actually needed.",
        DeprecationWarning,
        stacklevel=2,
    )

    from .spectral_response import compute_spectral_response  # deferred, see above

    self._get_fwhm_unbinned()

    sr = compute_spectral_response(
        band_centers_unbinned=self.wavelengths_unbinned,
        fwhm_unbinned=self.fwhm_unbinned,
        bin_factor=self.bin_factor,
        ssi_source="tsis",
        grid="uniform-1000",
        generate_figures=generate_figures,
    )

    self.spectral_response_csiro = sr

    self.csiro_srfs_csr = sr.srf_unbinned
    self.csiro_ssi = sr.ssi
    self.csiro_solar_wavelengths = sr.grid_wl
    self.csiro_binned_srfs = sr.srf.toarray()
    self.csiro_effective_fwhm = sr.effective_fwhm
    self.csiro_esun = sr.esun

    return


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


def bin_srf(srfs_csr, bin_factor, solar_wavelengths=None, generate_figures=False):
    """
    Bin neighboring SRFs by adding them element-wise.
    
    Parameters:
    -----------
    srfs_csr : scipy.sparse.csr_matrix
        Sparse CSR matrix of SRFs (n_bands times n_wavelengths)
    bin_factor : int
        Number of neighboring bands to bin together
    solar_wavelengths : array, optional
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
    
    for bin_idx in tqdm(range(n_bins), desc="Binning bands"):
        start_band = bin_idx * bin_factor
        end_band = start_band + bin_factor
        
        # Get the slice of bands for this bin
        band_slice = srfs_csr[start_band:end_band, :]
        
        # Sum the bands element-wise
        # .sum(axis=0) returns a matrix, .A1 flattens to 1D array
        binned_srf = band_slice.sum(axis=0).A1
        
        # Normalize
        srf_max = binned_srf.max()
        if srf_max > 0:
            binned_srf = binned_srf / srf_max

        # Store as sparse row
        binned_rows.append(csr_matrix(binned_srf))
        
        # Keep track of which original bands are in this bin
        bin_indices.append(list(range(start_band, end_band)))
        
        #if (bin_idx + 1) % 10 == 0:
        #    print(f"Processed {bin_idx + 1}/{n_bins} bins")
    
    # Stack all binned rows into a single CSR matrix
    binned_srfs_csr = vstack(binned_rows, format='csr')
    
    print(f"[INFO] Binned SRF matrix shape: {binned_srfs_csr.shape}")
    #print(f"Non-zero elements: {binned_srfs_csr.nnz}")
    #print(f"Average non-zero per binned band: {binned_srfs_csr.nnz / n_bins:.1f}")
    

    # Optional visualization
    if solar_wavelengths is not None and generate_figures:
        visualize_srf_binning(srfs_csr, binned_srfs_csr, bin_factor, 
                         solar_wavelengths, bin_indices)
    
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
    plt.savefig('1_hypso_srf_visualized.png')
    plt.close()
    
    return fig


def compute_esun_sparse_efficient(srfs_csr, ssi):
    """
    Most efficient version using sparse matrix operations.
    This avoids converting to dense arrays entirely.
    
    Parameters:
    -----------
    srfs_csr : scipy.sparse.csr_matrix
        CSR matrix of aligned SRFs
    ssi : numpy array
        Solar spectral irradiance
    
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
    Compute per-band Esun (SSI weighted by each band's SRF).

    Only the sparse implementation remains - the 'vectorized' and 'loop'
    variants were unused alternatives (every caller passed method='sparse')
    and were removed in the spectral-response cleanup; the `method` parameter
    is kept for signature compatibility and rejects anything else loudly
    rather than silently computing with a different algorithm.

    Returns:
    --------
    esun_list : list
        Esun values for all bands
    """
    if method != 'sparse':
        raise ValueError(
            f"compute_esun method {method!r} was removed - only 'sparse' remains "
            f"(the variant every caller used; see hypso.reflectance.spectral_response)."
        )
    return compute_esun_sparse_efficient(srfs_csr, ssi)


def compute_effective_fwhm(srfs_csr, solar_wavelengths):

    wavelengths = solar_wavelengths

    band_indices = range(srfs_csr.shape[0])
    
    effective_fwhm_array = np.zeros(len(band_indices))

    
    for i in band_indices:

        # Get the SRF for this band (as dense array)
        # .A1 flattens the matrix to 1D array
        srf = srfs_csr[i, :].toarray().flatten()
        
        max_idx = np.argmax(srf)
        max_srf = srf[max_idx]

        half_max = max_srf / 2

        indices = np.where(srf >= half_max)[0]

        if len(indices) > 0:
            lower_idx = indices[0]
            upper_idx = indices[-1]
            
            # Get corresponding x values
            lower_wl = wavelengths[lower_idx]
            upper_wl = wavelengths[upper_idx]
            
            # Calculate FWHM
            effective_fwhm = upper_wl - lower_wl

        else:
            effective_fwhm = 0

        effective_fwhm_array[i] = float(effective_fwhm)

    return effective_fwhm_array




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