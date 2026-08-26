"""SpectralResponse: one value object for HYPSO's spectral response state.

What this supersedes (see REFACTOR_PROGRESS.md's SRF assessment): the package
previously computed spectral response functions in two near-duplicate code
paths and scattered the results across two differently-named attribute
families on the capture object:

- ``compute_toa_reflectance`` built SRFs on the (truncated) native SSI grid
  and HypsoCapture stored them as ``srf``/``srf_ssi``/``srf_ssi_wl``/``esun``/
  ``esun_wl``/``effective_fwhm`` - names that hid what things were: ``srf``
  was the *binned* SRF matrix, and ``esun_wl`` was *binned band centers*, not
  an SSI wavelength grid (despite the name symmetry with ``srf_ssi_wl``).
- ``compute_csiro_srfs`` (the uniform-grid convolution variant - both this and
  the path above convolve a Gaussian SRF against solar irradiance; what
  differs is the wavelength grid the convolution runs on, not the operation
  itself) rebuilt ~the same thing on a fixed 1000-point 350-850 nm grid and
  stored it as
  ``csiro_srfs_csr`` (*unbinned*, sparse)/``csiro_binned_srfs`` (binned,
  *dense*)/``csiro_ssi``/``csiro_solar_wavelengths``/``csiro_effective_fwhm``/
  ``csiro_esun``.

Here, "binned/unbinned/which grid/which SSI" are explicit *fields and
parameters* of one dataclass built by one function, instead of naming
conventions spread across a dozen loose attributes. ``compute_spectral_
response(grid="native-truncated")`` reproduces the first path exactly;
``grid="uniform-1000"`` reproduces the CSIRO path exactly (same math, moved -
verified against reference outputs captured from the pre-refactor code).

The old ``compute_toa_reflectance`` entry point still works - it's now a thin
wrapper over this module, and HypsoCapture keeps populating its legacy
attributes - because the Polymer connector
(hypso.ac.adapters.polymer's generate_srf_nc/ssi/esun) and the L1D metadata
writer still read them, and per the current plan the AC connectors get
migrated to read ``satobj.spectral_response`` directly in a later, separate
pass (together with the eoread/ACOLITE reader updates). The generated SRF
NetCDF format is FROZEN either way: Polymer resolves and reads it through
eotools' get_SRF (Band_<n> variables on wav_Band_<n> coords), so nothing
about that file may change until Polymer-side code is updated in that later
pass.

``compute_csiro_srfs`` itself (the uniform-grid convolution entry point that
used to populate the ``csiro_*`` attribute family) has been removed outright -
confirmed zero remaining callers after hypso-processing-pipeline dropped its
call (2026-08-25) and hypso-ac-processing (its only other caller) was
confirmed fully superseded (2026-08-26). ``compute_spectral_response(
grid="uniform-1000")`` below is its direct replacement for anyone who still
needs that exact grid/SSI combination.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from .toa_reflectance import (
    compute_srf,
    bin_srf,
    bin_sensor_wavelengths,
    compute_esun,
    compute_effective_fwhm,
    load_ssi,
    load_thuillier_ssi,
)


@dataclass(frozen=True)
class SpectralResponse:
    """Everything derived from "a set of Gaussian band responses sampled on a
    solar-spectral-irradiance wavelength grid", in one place.

    Shapes: n_bands = binned band count, n_unbinned = pre-binning band count,
    n_grid = wavelength grid length.
    """
    #: Band-center wavelengths (nm) of the (binned) bands - (n_bands,).
    band_centers: np.ndarray
    #: FWHM (nm) of the *unbinned* Gaussians the SRFs were built from -
    #: (n_unbinned,).
    fwhm: np.ndarray
    #: Binned SRF matrix, sparse - (n_bands, n_grid). This is what the legacy
    #: ``satobj.srf`` attribute held.
    srf: csr_matrix
    #: Pre-binning SRF matrix, sparse - (n_unbinned, n_grid). This is what the
    #: legacy ``satobj.csiro_srfs_csr`` attribute held.
    srf_unbinned: csr_matrix
    #: Wavelength grid (nm) the SRFs and SSI are sampled on - (n_grid,).
    grid_wl: np.ndarray
    #: Solar spectral irradiance on grid_wl - (n_grid,).
    ssi: np.ndarray
    #: Per-band solar irradiance (SSI weighted by each band's SRF) - (n_bands,).
    esun: np.ndarray
    #: Per-band effective FWHM computed from the binned SRFs - (n_bands,).
    effective_fwhm: np.ndarray
    #: Per-unbinned-band effective FWHM, computed the same way as
    #: effective_fwhm but from srf_unbinned instead of srf - (n_unbinned,).
    #: Not the same thing as the `fwhm` field above: `fwhm` is the nominal
    #: value used as an *input* to build each unbinned Gaussian SRF;
    #: this is measured empirically from that SRF's own half-max width,
    #: the same way effective_fwhm is measured from the binned one.
    effective_fwhm_unbinned: np.ndarray
    #: Spectral binning factor the srf/esun/effective_fwhm reflect.
    bin_factor: int
    #: Which SSI: "tsis" (TSIS-1 HSRS) or "thuillier" (Thuillier 2002).
    ssi_source: str
    #: Which grid: "native-truncated" (SSI's own grid, truncated to the sensor
    #: range +/- 3 sigma) or "uniform-1000" (1000 points over 350-850 nm, the
    #: uniform-grid convolution convention).
    grid: str


def compute_spectral_response(band_centers_unbinned,
                              fwhm_unbinned,
                              bin_factor: int,
                              ssi_source: str = "tsis",
                              grid: str = "native-truncated",
                              generate_figures: bool = False) -> SpectralResponse:
    """Build a SpectralResponse from unbinned band centers + FWHM.

    grid="native-truncated" reproduces exactly what compute_toa_reflectance's
    inline SRF block computed; grid="uniform-1000" reproduces exactly what the
    old compute_csiro_srfs computed (which always used the TSIS SSI). Same
    code, same call order - relocated, not rewritten.
    """
    if ssi_source == "thuillier":
        ssi, solar_wavelengths = load_thuillier_ssi()
    elif ssi_source == "tsis":
        ssi, solar_wavelengths = load_ssi()
    else:
        raise ValueError(f"Unknown ssi_source: {ssi_source!r} (use 'tsis' or 'thuillier')")

    if grid == "native-truncated":
        srfs_csr, grid_ssi, grid_wl = compute_srf(
            ssi=ssi,
            solar_wavelengths=solar_wavelengths,
            sensor_wavelengths=band_centers_unbinned,
            sensor_fwhm=fwhm_unbinned,
        )

    elif grid == "uniform-1000":
        # The uniform-grid convolution variant: SSI interpolated onto 1000
        # uniform points over 350-850 nm, SRFs built on that full grid with no
        # truncation. Body relocated verbatim from the old compute_csiro_srfs.
        grid_wl = np.linspace(350, 850, 1000)
        grid_ssi = np.interp(grid_wl, solar_wavelengths, ssi)

        sensor_wavelengths = band_centers_unbinned
        sigma_nm = fwhm_unbinned / (2 * np.sqrt(2 * np.log(2)))

        sensor_band_indices = [np.abs(grid_wl - w).argmin() for w in sensor_wavelengths]

        n_bands = len(sensor_wavelengths)
        aligned_srfs_sparse = lil_matrix((n_bands, len(grid_wl)), dtype=np.float32)

        for i, (adjusted_idx, center_wavelength) in enumerate(zip(sensor_band_indices, sensor_wavelengths)):
            start_lambda = center_wavelength - (3 * sigma_nm[i])
            end_lambda = center_wavelength + (3 * sigma_nm[i])

            start_idx = np.abs(grid_wl - start_lambda).argmin()
            end_idx = np.abs(grid_wl - end_lambda).argmin()

            srf_wavelengths = grid_wl[start_idx:end_idx + 1]

            gx = np.linspace(-3 * sigma_nm[i], 3 * sigma_nm[i], len(srf_wavelengths))
            gaussian_srf = np.exp(-(gx / sigma_nm[i]) ** 2 / 2)

            aligned_srfs_sparse[i, start_idx:end_idx + 1] = gaussian_srf

        srfs_csr = aligned_srfs_sparse.tocsr()

    else:
        raise ValueError(f"Unknown grid: {grid!r} (use 'native-truncated' or 'uniform-1000')")

    binned_band_centers = bin_sensor_wavelengths(band_centers_unbinned, bin_factor)
    binned_srfs_csr = bin_srf(srfs_csr, bin_factor, grid_wl, generate_figures)
    esun = np.array(compute_esun(srfs_csr=binned_srfs_csr, ssi=grid_ssi, method="sparse"))
    effective_fwhm = compute_effective_fwhm(srfs_csr=binned_srfs_csr, solar_wavelengths=grid_wl)
    effective_fwhm_unbinned = compute_effective_fwhm(srfs_csr=srfs_csr, solar_wavelengths=grid_wl)

    return SpectralResponse(
        band_centers=np.asarray(binned_band_centers),
        fwhm=np.asarray(fwhm_unbinned),
        srf=binned_srfs_csr,
        srf_unbinned=srfs_csr,
        grid_wl=np.asarray(grid_wl),
        ssi=np.asarray(grid_ssi),
        esun=esun,
        effective_fwhm=np.asarray(effective_fwhm),
        effective_fwhm_unbinned=np.asarray(effective_fwhm_unbinned),
        bin_factor=bin_factor,
        ssi_source=ssi_source,
        grid=grid,
    )
