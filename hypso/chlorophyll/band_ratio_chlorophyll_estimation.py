import numpy as np



def run_band_ratio_chlorophyll_estimation(cube: np.ndarray,
                                          mask: np.ndarray, 
                                          wavelengths: np.ndarray,
                                          spatial_dimensions: tuple[int, int],
                                          factor: float = 0.1
                                          ) -> np.ndarray:

    if mask is None:
        mask = np.zeros(spatial_dimensions, dtype=bool)
    
    if factor is None:
        factor = 0.1

    numerator_wavelength = 549
    denominator_wavelength = 663

    a = abs(wavelengths - numerator_wavelength)
    numerator_index = np.argmin(a)
    
    a = abs(wavelengths - denominator_wavelength)
    denominator_index = np.argmin(a)

    #print(wavelengths[numerator_index])
    #print(wavelengths[denominator_index])

    chl = cube[:,:,numerator_index] / cube[:,:,denominator_index]

    chl = np.ma.masked_array(chl, mask, fill_value=np.nan)

    #factor = 0.88
    #factor = 0.1

    try:
        # Only get maximum from unmasked data
        chl_compressed = chl.compressed()
        chl_maximum = chl_compressed.max()
        offset = factor * chl_maximum
        chl = chl - offset
        chl[chl < 0] = 0
    except ValueError:
        chl = np.zeros(spatial_dimensions)
        chl = np.ma.masked_array(chl, mask, fill_value=np.nan)

    #chl = chl[:,::-1]

    return chl