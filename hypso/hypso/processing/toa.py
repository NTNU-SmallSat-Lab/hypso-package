"""L1C -> L1D: TOA radiance to TOA reflectance.

The L1D tree also carries metadata/srf (per-band SRFs, esun, Earth-Sun
distance, hypso.srf.srf_dataset) for readers that convert radiance
themselves, e.g. Polymer. The spectral response and solar spectra are in
hypso.srf."""
import numpy as np
import xarray as xr

from hypso.io.product import product_name
from hypso.io.provenance import add_step
from hypso.srf import compute_spectral_response, earth_sun_distance, srf_dataset

from .capture import capture_params


def toa_reflectance(radiance: np.ndarray, esun, iso_time, solar_zenith) -> np.ndarray:
    """TOA reflectance from TOA radiance (lines, samples, bands):
    rho = pi L d^2 / (F0 cos(sza)), with F0 = esun per band (mW m-2 nm-1,
    numerically equal to the radiance's W m-2 um-1), d the Earth-Sun
    distance in AU and sza the solar zenith angle (degrees). "Normalized
    reflectances": https://oceanopticsbook.info/view/atmospheric-correction/normalized-reflectances
    """
    distance = earth_sun_distance(iso_time)
    multiplier = (np.asarray(esun)[None, None, :] * np.cos(np.radians(solar_zenith))[:, :, None]) \
        / (np.pi * distance ** 2)
    return (radiance / multiplier).astype(np.asarray(radiance).dtype)


def spectral_response(tree: xr.DataTree, ssi_source: str = "tsis", use_unbinned: bool = True):
    """The SpectralResponse (SRFs, solar irradiance, effective FWHM) of a
    calibrated tree's bands - what L1D reflectance and metadata/srf use."""
    params = capture_params(tree)
    if use_unbinned:
        unbinned = tree["metadata/corrections"].to_dataset(inherit=False)["wavelengths_unbinned"].values
        band_centers, band_fwhm, bin_factor = unbinned, params.fwhm(unbinned), params.bin_factor
    else:
        wavelengths = tree.dataset["radiation_wavelength"].values
        band_centers, band_fwhm, bin_factor = wavelengths, params.fwhm(wavelengths), 1
    return compute_spectral_response(band_centers_unbinned=band_centers, fwhm_unbinned=band_fwhm,
                                     bin_factor=bin_factor, ssi_source=ssi_source)


def srf_node(tree: xr.DataTree, ssi_source: str = "tsis", use_unbinned: bool = True) -> xr.Dataset:
    """metadata/srf for a calibrated tree (computed, not read from it)."""
    wavelengths = tree.dataset["radiation_wavelength"].values
    params = capture_params(tree)
    return srf_dataset(spectral_response(tree, ssi_source, use_unbinned), wavelengths, params.fwhm(wavelengths),
                       params.iso_time)


def l1c_to_l1d(tree: xr.DataTree, use_direct_georef: bool = False, ssi_source: str = "tsis",
               use_unbinned: bool = True) -> xr.DataTree:
    """TOA reflectance `rhot` from an L1C tree.

    use_direct_georef: use solar_zenith_direct (direct georeferencing)
        instead of solar_zenith, when present.
    ssi_source: "tsis" (TSIS-1 HSRS) or "thuillier".
    use_unbinned: build SRFs from the unbinned wavelengths and bin them
        (default), instead of from the binned band centres.
    """
    if tree.attrs.get("processing_level") != "L1C":
        raise ValueError(f"l1c_to_l1d needs an L1C tree, got {tree.attrs.get('processing_level')}")
    params = capture_params(tree)
    root = tree.to_dataset(inherit=False)
    radiance = root[product_name(tree)]
    wavelengths = root["radiation_wavelength"].values
    fwhm = params.fwhm(wavelengths)

    zenith_name = "solar_zenith_direct" if use_direct_georef and "solar_zenith_direct" in root else "solar_zenith"
    sr = spectral_response(tree, ssi_source=ssi_source, use_unbinned=use_unbinned)
    reflectance = toa_reflectance(radiance.values, sr.esun, params.iso_time, root[zenith_name].values)

    out = root.drop_vars([product_name(tree), "wavelength", "radiation_wavelength", "fwhm"])
    out["rhot"] = xr.DataArray(reflectance, dims=radiance.dims, coords={
        "wavelength": ("bands", np.around(wavelengths, 1)),
        "radiation_wavelength": ("bands", wavelengths),
        "fwhm": ("bands", fwhm)})
    out.attrs["processing_level"] = "L1D"

    result = tree.copy()
    result.dataset = out
    result["metadata/srf"] = xr.DataTree(srf_dataset(sr, wavelengths, fwhm, params.iso_time))
    source = tree.encoding.get("source")
    result.encoding.pop("source", None)
    add_step(result, "l1d", input_file=source or "L1C (in memory)",
             parameters={"ssi_source": ssi_source, "srf_grid": sr.grid, "bin_factor": int(sr.bin_factor),
                         "use_unbinned": use_unbinned, "solar_zenith": zenith_name,
                         "earth_sun_distance_au": earth_sun_distance(params.iso_time)})
    return result
