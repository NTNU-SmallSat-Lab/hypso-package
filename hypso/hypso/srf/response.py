"""Spectral response of HYPSO's bands: Gaussian spectral response functions
(SRFs) per unbinned band, sampled on the solar irradiance spectrum's
wavelength grid and binned like the image; and what follows from them -
per-band solar irradiance (esun, F0) and effective FWHM.

compute_spectral_response() builds all of it into one SpectralResponse;
srf_dataset() turns that into a product's metadata/srf group.
"""
import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix, lil_matrix, vstack

from .solar import earth_sun_distance, load_solar_irradiance

logger = logging.getLogger(__name__)

# How each solar spectrum is named in metadata (until 2026-10-26 the esun
# description always said TSIS-1, even for Thuillier).
SOLAR_SPECTRA = {"tsis": "TSIS-1 HSRS v2 (2022-11-30)", "thuillier": "Thuillier et al. (2002) spectrum"}


@dataclass(frozen=True)
class SpectralResponse:
    """SRFs and what is derived from them. Shapes: n_bands = binned band
    count, n_unbinned = pre-binning band count, n_grid = wavelength grid."""
    #: Band-centre wavelengths (nm) of the binned bands - (n_bands,).
    band_centers: np.ndarray
    #: FWHM (nm) of the unbinned Gaussians the SRFs were built from - (n_unbinned,).
    fwhm: np.ndarray
    #: Binned SRF matrix, sparse - (n_bands, n_grid).
    srf: csr_matrix
    #: Unbinned SRF matrix, sparse - (n_unbinned, n_grid).
    srf_unbinned: csr_matrix
    #: Wavelength grid (nm) the SRFs and irradiance are sampled on - (n_grid,).
    grid_wl: np.ndarray
    #: Solar spectral irradiance on grid_wl, mW m-2 nm-1 - (n_grid,).
    ssi: np.ndarray
    #: Per-band solar irradiance F0 (irradiance weighted by each band's SRF) - (n_bands,).
    esun: np.ndarray
    #: Per-band effective FWHM, measured from the binned SRFs - (n_bands,).
    effective_fwhm: np.ndarray
    #: Per-unbinned-band effective FWHM, measured from srf_unbinned (not the
    #: same as `fwhm`, the nominal input) - (n_unbinned,).
    effective_fwhm_unbinned: np.ndarray
    #: Spectral binning factor the binned fields reflect.
    bin_factor: int
    #: "tsis" (TSIS-1 HSRS) or "thuillier" (Thuillier 2002).
    ssi_source: str
    #: Wavelength grid: the irradiance spectrum's own, truncated to the bands'
    #: range +/- 3 sigma.
    grid: str = "native-truncated"


def compute_spectral_response(band_centers_unbinned, fwhm_unbinned, bin_factor: int,
                              ssi_source: str = "tsis") -> SpectralResponse:
    """Spectral response of a sensor's bands.

    Parameters
    ----------
    band_centers_unbinned, fwhm_unbinned : array_like
        Band centres and FWHM in nm before spectral binning.
    bin_factor : int
        Spectral binning factor of the image.
    ssi_source : {"tsis", "thuillier"}, default "tsis"
        Solar irradiance spectrum for the per-band F0.

    Returns
    -------
    SpectralResponse
    """
    ssi, solar_wavelengths = load_solar_irradiance(ssi_source)
    srfs, grid_ssi, grid_wl = gaussian_srfs(ssi, solar_wavelengths, band_centers_unbinned, fwhm_unbinned)
    binned = bin_srfs(srfs, bin_factor)
    return SpectralResponse(
        band_centers=np.asarray(bin_band_centers(band_centers_unbinned, bin_factor)),
        fwhm=np.asarray(fwhm_unbinned),
        srf=binned,
        srf_unbinned=srfs,
        grid_wl=np.asarray(grid_wl),
        ssi=np.asarray(grid_ssi),
        esun=band_irradiance(binned, grid_ssi),
        effective_fwhm=effective_fwhm(binned, grid_wl),
        effective_fwhm_unbinned=effective_fwhm(srfs, grid_wl),
        bin_factor=bin_factor,
        ssi_source=ssi_source,
    )


def gaussian_srfs(ssi, solar_wavelengths, sensor_wavelengths, sensor_fwhm):
    """One Gaussian SRF (peak 1, +/- 3 sigma) per band on the irradiance
    spectrum's wavelength grid, truncated to the first band's start and the
    last band's end. Returns (srfs as sparse CSR, truncated irradiance,
    truncated wavelengths)."""
    sigma_nm = sensor_fwhm / (2 * np.sqrt(2 * np.log(2)))

    sensor_band_indices = [np.abs(solar_wavelengths - w).argmin() for w in sensor_wavelengths]
    first_band_start = solar_wavelengths[sensor_band_indices[0]] - 3 * sigma_nm[0]
    last_band_end = solar_wavelengths[sensor_band_indices[-1]] + 3 * sigma_nm[-1]
    start = np.abs(solar_wavelengths - first_band_start).argmin()
    end = np.abs(solar_wavelengths - last_band_end).argmin()
    truncated_ssi = ssi[start:end + 1]
    truncated_wavelengths = solar_wavelengths[start:end + 1]

    srfs = lil_matrix((len(sensor_wavelengths), len(truncated_wavelengths)), dtype=np.float32)
    for i, center in enumerate(sensor_wavelengths):
        start_idx = np.abs(truncated_wavelengths - (center - 3 * sigma_nm[i])).argmin()
        end_idx = np.abs(truncated_wavelengths - (center + 3 * sigma_nm[i])).argmin()
        gx = np.linspace(-3 * sigma_nm[i], 3 * sigma_nm[i], end_idx - start_idx + 1)
        srfs[i, start_idx:end_idx + 1] = np.exp(-(gx / sigma_nm[i]) ** 2 / 2)
    return srfs.tocsr(), truncated_ssi, truncated_wavelengths


def bin_band_centers(band_centers, bin_factor):
    return np.asarray(band_centers).reshape(-1, bin_factor).mean(axis=1).reshape(-1)


def bin_srfs(srfs: csr_matrix, bin_factor: int) -> csr_matrix:
    """Sum each group of bin_factor neighbouring SRFs and renormalise to a
    peak of 1 (extra bands beyond a whole number of bins are dropped)."""
    n_bands = srfs.shape[0]
    n_bins = n_bands // bin_factor
    if n_bands % bin_factor:
        logger.warning("%d bands not evenly divisible by bin_factor %d; using the first %d",
                       n_bands, bin_factor, n_bins * bin_factor)
        srfs = srfs[:n_bins * bin_factor, :]
    rows = []
    for b in range(n_bins):
        binned = srfs[b * bin_factor:(b + 1) * bin_factor, :].sum(axis=0).A1
        if binned.max() > 0:
            binned = binned / binned.max()
        rows.append(csr_matrix(binned))
    return vstack(rows, format="csr")


def band_irradiance(srfs: csr_matrix, ssi) -> np.ndarray:
    """Per-band solar irradiance (F0, esun): the irradiance averaged with
    each band's SRF as weights."""
    esun = np.zeros(srfs.shape[0])
    for i in range(srfs.shape[0]):
        start, end = srfs.indptr[i], srfs.indptr[i + 1]
        if start < end:
            weights = srfs.data[start:end]
            esun[i] = np.sum(ssi[srfs.indices[start:end]] * (weights / np.sum(weights)))
    return esun


def effective_fwhm(srfs: csr_matrix, wavelengths) -> np.ndarray:
    """Width (nm) of each SRF at half its maximum."""
    widths = np.zeros(srfs.shape[0])
    for i in range(srfs.shape[0]):
        srf = srfs[i, :].toarray().flatten()
        above = np.where(srf >= srf[np.argmax(srf)] / 2)[0]
        widths[i] = float(wavelengths[above[-1]] - wavelengths[above[0]]) if len(above) else 0.0
    return widths


def srf_dataset(sr: SpectralResponse, wavelengths, fwhm, iso_time=None) -> xr.Dataset:
    """A product's ``metadata/srf`` group.

    Parameters
    ----------
    sr : SpectralResponse
    wavelengths, fwhm : array_like
        The product's band centres and FWHM (nm).
    iso_time : str, optional
        Acquisition time, for ``earth_sun_distance_au``.

    Returns
    -------
    xarray.Dataset
        ``esun``, ``esun_wavelengths``, ``effective_fwhm(_unbinned)``, and one
        ``Band_<i>`` variable per band (the non-zero part of its binned SRF,
        on its own ``wav_Band_<i>`` coordinate). Polymer reads its SRFs from
        here (:func:`hypso.ac.ac_polymer.ac_polymer_srf_getter_from_l1`).
    """
    data, attrs = {}, {}
    for name, value in (("esun", sr.esun), ("esun_wavelengths", sr.band_centers),
                        ("effective_fwhm", sr.effective_fwhm),
                        ("effective_fwhm_unbinned", sr.effective_fwhm_unbinned)):
        data[name] = xr.Variable((name,), np.asarray(value, dtype="f8"))
    data["esun"].attrs = {
        "units": "mW m-2 nm-1",
        "description": (f"Band-averaged (SRF-weighted) solar irradiance F0 at 1 AU, from the "
                        f"{SOLAR_SPECTRA[sr.ssi_source]}; the F0 used for this file's TOA reflectance")}
    if iso_time is not None:
        attrs["earth_sun_distance_au"] = earth_sun_distance(iso_time)

    grid = np.asarray(sr.grid_wl)
    for index, wavelength in enumerate(wavelengths):
        band = f"Band_{index}"
        response = sr.srf[index, :].toarray().flatten()
        nonzero = response > 0
        values, wl = (response[nonzero], grid[nonzero]) if np.any(nonzero) else (response, grid)
        dim = f"wav_{band}"
        data[dim] = xr.Variable((dim,), wl, {"units": "nm"})
        data[band] = xr.Variable((dim,), values, {
            "band_info": band, "band_wavelength": float(wavelength), "index": index,
            "effective_fwhm": float(sr.effective_fwhm[index]), "center_fwhm": float(fwhm[index])})
    return xr.Dataset(data, attrs=attrs)
