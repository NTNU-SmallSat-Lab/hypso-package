"""Spectral response of HYPSO's bands (SRFs, per-band solar irradiance,
effective FWHM) and the reference solar spectra and Earth-Sun distance it
uses. The TOA reflectance step itself is hypso.processing.toa."""
from .solar import earth_sun_distance, load_solar_irradiance
from .response import (SpectralResponse, band_irradiance, bin_band_centers, bin_srfs, compute_spectral_response,
                       effective_fwhm, gaussian_srfs, srf_dataset)
