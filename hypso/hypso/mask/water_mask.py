from ..spectral_analysis import get_closest_wavelength_index
import numpy as np
from skimage.filters import threshold_otsu














def _simple_water_mask(green, nir, blue=None, red=None):
    """
    Simple but effective water masking rules
    """
    # Rule 1: Basic NDWI threshold
    ndwi = (green - nir) / (green + nir + 1e-10)
    mask1 = ndwi > -0.25
    
    # Rule 2: Absolute reflectance thresholds
    mask2 = (nir < 0.15) & (green > 0.08) & (green < 0.4)
    
    # Rule 3: If blue band available - water has high blue reflectance
    if blue is not None:
        mask3 = blue > 0.05
    else:
        mask3 = np.ones_like(green, dtype=bool)
    
    # Rule 4: If red band available - exclude bright red areas
    if red is not None:
        mask4 = red < 0.2
    else:
        mask4 = np.ones_like(green, dtype=bool)
    
    # Combine all rules
    water_mask = mask1 & mask2 & mask3 & mask4
    
    return water_mask, ndwi


def _robust_water_mask(blue, green, red, nir):
    """
    Robust water masking using multiple criteria
    """
    # Step 1: Calculate multiple indices
    ndwi = (green - nir) / (green + nir + 1e-10)
    ndvi = (nir - red) / (nir + red + 1e-10)  # To exclude vegetation
    
    # Step 2: Water spectral characteristics
    # 1. High NDWI
    high_ndwi = ndwi > -0.2
    
    # 2. Low NIR reflectance (water absorbs NIR)
    low_nir = nir < 0.15
    
    # 3. Moderate Green reflectance
    moderate_green = (green > 0.05) & (green < 0.3)
    
    # 4. Not vegetation (low NDVI)
    not_vegetation = ndvi < 0.2
    
    # 5. Blue-Green dominance
    blue_green_dominant = (green + blue) > (red + nir)
    
    # Step 3: Combine rules
    water_mask = (
        high_ndwi & 
        low_nir & 
        moderate_green & 
        not_vegetation & 
        blue_green_dominant
    )
    
    # Step 4: Clean up with morphological operations
    from scipy import ndimage
    water_mask = ndimage.binary_opening(water_mask)  # Remove small noise
    water_mask = ndimage.binary_closing(water_mask)  # Fill small holes
    
    return water_mask, ndwi


def _automated_water_detection_mask(green, nir, blue=None):
    """
    Automated water detection without fixed NDWI threshold
    """
    # Calculate NDWI
    ndwi = (green - nir) / (green + nir + 1e-10)
    
    # Flatten arrays for processing
    ndwi_flat = ndwi.flatten()
    green_flat = green.flatten()
    nir_flat = nir.flatten()
    
    # Remove NaN values for threshold calculations
    valid_mask = ~np.isnan(ndwi_flat)
    ndwi_valid = ndwi_flat[valid_mask]
    
    # Method 1: Otsu's automatic threshold
    try:
        thresh_otsu = threshold_otsu(ndwi_valid)
        water_mask_otsu_flat = ndwi_flat > thresh_otsu
    except:
        thresh_otsu = -0.2
        water_mask_otsu_flat = ndwi_flat > thresh_otsu
    
    # Method 2: Percentile-based (water is usually in higher NDWI percentiles)
    p85 = np.percentile(ndwi_valid, 85)
    water_mask_percentile_flat = ndwi_flat > p85
    
    # Method 3: Rule-based with multiple bands
    if blue is not None:
        blue_flat = blue.flatten()
        # Water: High Green, Low NIR, High Blue
        rule_mask_flat = (green_flat > 0.1) & (nir_flat < 0.2) & (blue_flat > 0.05) & (ndwi_flat > -0.3)
    else:
        rule_mask_flat = (green_flat > 0.1) & (nir_flat < 0.2) & (ndwi_flat > -0.3)
    
    # Combine methods
    final_mask_flat = water_mask_otsu_flat | water_mask_percentile_flat | rule_mask_flat
    
    # Reshape back to original dimensions
    final_mask = final_mask_flat.reshape(ndwi.shape)
    
    return final_mask, ndwi


def _get_water_mask_bands(satobj):

    blue_idx = get_closest_wavelength_index(satobj=satobj, wavelength=475)
    green_idx = get_closest_wavelength_index(satobj=satobj, wavelength=560)
    red_idx = get_closest_wavelength_index(satobj=satobj, wavelength=650)
    nir_idx = get_closest_wavelength_index(satobj=satobj, wavelength=780)

    #blue = satobj.l1d_cube[:,:,27].to_numpy() #450-500nm
    #green = satobj.l1d_cube[:,:,52].to_numpy() #540-580nm
    #red = satobj.l1d_cube[:,:,78].to_numpy() #620-680nm
    #nir = satobj.l1d_cube[:,:,116].to_numpy() #760-800nm

    blue = satobj.l1d_cube[:,:,blue_idx].to_numpy() #450-500nm
    green = satobj.l1d_cube[:,:,green_idx].to_numpy() #540-580nm
    red = satobj.l1d_cube[:,:,red_idx].to_numpy() #620-680nm
    nir = satobj.l1d_cube[:,:,nir_idx].to_numpy() #760-800nm

    return blue, green, red, nir


def simple_water_mask(satobj):

    blue, green, red, nir = _get_water_mask_bands(satobj=satobj)

    mask, ndwi = _simple_water_mask(green=green, nir=nir, blue=blue, red=red)

    return mask, ndwi


def robust_water_mask(satobj):

    blue, green, red, nir = _get_water_mask_bands(satobj=satobj)

    mask, ndwi = _robust_water_mask(blue=blue, green=green, red=red, nir=nir)

    return mask, ndwi


def automated_water_detection_mask(satobj):

    blue, green, red, nir = _get_water_mask_bands(satobj=satobj)

    mask, ndwi = _automated_water_detection_mask(green=green, nir=nir, blue=blue)

    return mask, ndwi



