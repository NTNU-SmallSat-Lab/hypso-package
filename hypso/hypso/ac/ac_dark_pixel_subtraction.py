import numpy as np


def ac_dark_pixel_subtraction(self, method='min', VERBOSE=True):

    key = "Rrs"

    l1d_cube = self.l1d_cube

    corrected_cube, dark_spectrum = dark_pixel_subtraction_per_band(datacube=l1d_cube, method=method)

    self.l2a_cube["dps"] = corrected_cube
    self.l2a_cube["dps"].attrs['l2_variable_name'] = key


    return dark_spectrum


def dark_pixel_subtraction_per_band(datacube, method='min', percentile=None):
    """
    Perform dark pixel subtraction by finding the darkest pixel(s) per band.
    
    Parameters:
    -----------
    datacube : numpy.ndarray
        Shape (x, y, bands)
    method : str
        'min' for absolute minimum, 'percentile' for percentile-based dark value
    percentile : float
        If method='percentile', use this percentile (e.g., 1 for 1st percentile)
    
    Returns:
    --------
    corrected_cube : numpy.ndarray
        Shape (x, y, bands)
    dark_spectrum : numpy.ndarray
        The synthetic dark spectrum used for subtraction, shape (bands,)
    """
    # Find the darkest value for each band
    if method == 'min':
        # Take the absolute minimum across spatial dimensions
        dark_spectrum = np.min(datacube, axis=(0, 1))  # Shape: (bands,)
    
    elif method == 'percentile':
        # Use a percentile to avoid single-pixel noise
        if percentile is None:
            percentile = 1  # Default to 1st percentile
        dark_spectrum = np.percentile(datacube, percentile, axis=(0, 1))
    
    else:
        raise ValueError("Method must be 'min' or 'percentile'")
    
    #print(dark_spectrum.shape)
    # Subtract dark spectrum from all pixels
    # Broadcast dark_spectrum to match datacube shape
    corrected_cube = datacube - dark_spectrum
    
    # Clip negative values to zero
    corrected_cube = np.clip(corrected_cube, 0, None)
    
    return corrected_cube, dark_spectrum

