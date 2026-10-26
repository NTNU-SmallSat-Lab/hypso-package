"""L1C to L1D: top-of-atmosphere radiance to reflectance.

The L1D tree also carries ``metadata/srf`` (per-band spectral response
functions, solar irradiance and Earth-Sun distance, from
:func:`hypso.srf.srf_dataset`) for readers that convert radiance
themselves, such as Polymer. The spectral response and solar spectra are in
:mod:`hypso.srf`.
"""
import numpy as np
import xarray as xr

from hypso.io.product import product_name
from hypso.io.provenance import add_step
from hypso.srf import compute_spectral_response, earth_sun_distance, srf_dataset

from .capture import capture_params


def toa_reflectance(radiance: np.ndarray, esun, iso_time, solar_zenith) -> np.ndarray:
    """Top-of-atmosphere reflectance from radiance.

    ``rho = pi L d^2 / (F0 cos(sza))``, with ``d`` the Earth-Sun distance in
    AU (`Normalized reflectances
    <https://oceanopticsbook.info/view/atmospheric-correction/normalized-reflectances>`_).

    Parameters
    ----------
    radiance : numpy.ndarray
        ``(lines, samples, bands)`` radiance in W m-2 um-1 sr-1.
    esun : array_like
        Per-band solar irradiance F0 in mW m-2 nm-1 (numerically equal to
        W m-2 um-1).
    iso_time : str
        Acquisition time (ISO 8601), for the Earth-Sun distance.
    solar_zenith : numpy.ndarray
        ``(lines, samples)`` solar zenith angle in degrees.

    Returns
    -------
    numpy.ndarray
        Reflectance, in the radiance's dtype.
    """
    distance = earth_sun_distance(iso_time)
    multiplier = (np.asarray(esun)[None, None, :] * np.cos(np.radians(solar_zenith))[:, :, None]) \
        / (np.pi * distance ** 2)
    return (radiance / multiplier).astype(np.asarray(radiance).dtype)


def spectral_response(tree: xr.DataTree, ssi_source: str = "tsis", use_unbinned: bool = True):
    """The :class:`~hypso.srf.SpectralResponse` of a calibrated tree's bands.

    Spectral response functions, solar irradiance and effective FWHM: what
    L1D reflectance and ``metadata/srf`` use.
    """
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
    """The ``metadata/srf`` group for a calibrated tree, computed (not read from it)."""
    wavelengths = tree.dataset["radiation_wavelength"].values
    params = capture_params(tree)
    return srf_dataset(spectral_response(tree, ssi_source, use_unbinned), wavelengths, params.fwhm(wavelengths),
                       params.iso_time)


def l1c_to_l1d(tree: xr.DataTree, use_direct_georef: bool = False, ssi_source: str = "tsis",
               use_unbinned: bool = True) -> xr.DataTree:
    """Top-of-atmosphere reflectance ``rhot`` from an L1C tree.

    Parameters
    ----------
    tree : xarray.DataTree
        L1C product tree (sensor geometry or gridded).
    use_direct_georef : bool, default False
        Use ``solar_zenith_direct`` instead of ``solar_zenith``, when present.
    ssi_source : {"tsis", "thuillier"}, default "tsis"
        Solar irradiance spectrum: TSIS-1 HSRS or Thuillier (2002).
    use_unbinned : bool, default True
        Build the spectral response functions from the unbinned wavelengths
        and bin them, instead of from the binned band centres.

    Returns
    -------
    xarray.DataTree
        L1D tree with ``metadata/srf``.
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
