"""L1C -> L1D: TOA radiance to TOA reflectance.

Same spectral response (SRFs, solar irradiance, effective FWHM) and
reflectance math as HypsoCapture._generate_l1d_cube_impl, taking and
returning a product tree. The L1D tree also carries metadata/srf (per-band
SRFs, esun, Earth-Sun distance) for readers that convert radiance
themselves, e.g. Polymer."""
import numpy as np
import xarray as xr

from hypso.io.product import PRODUCT_DIMS, product_name
from hypso.io.provenance import add_step
from hypso.reflectance import compute_reflectance, compute_spectral_response, earth_sun_distance

from .capture import capture_params


def srf_dataset(sr, wavelengths, fwhm, iso_time=None) -> xr.Dataset:
    """metadata/srf for a SpectralResponse `sr`: esun / esun_wavelengths /
    effective_fwhm(_unbinned), the Earth-Sun distance, and one Band_<i>
    variable per band (non-zero part of its binned SRF, on its own
    wav_Band_<i> wavelength coordinate). Same content as
    hypso.write.metadata_srf_group_writer."""
    data, attrs = {}, {}
    for name, value in (("esun", sr.esun), ("esun_wavelengths", sr.band_centers),
                        ("effective_fwhm", sr.effective_fwhm),
                        ("effective_fwhm_unbinned", sr.effective_fwhm_unbinned)):
        if value is not None:
            data[name] = xr.Variable((name,), np.asarray(value, dtype="f8"))
    if "esun" in data:
        data["esun"].attrs = {
            "units": "mW m-2 nm-1",
            "description": ("Band-averaged (SRF-weighted) solar irradiance F0 at 1 AU, from the "
                            "TSIS-1 HSRS v2 (2022-11-30); the F0 used for this file's TOA reflectance")}
        if iso_time is not None:
            attrs["earth_sun_distance_au"] = earth_sun_distance(iso_time)

    if sr.srf is not None and sr.grid_wl is not None:
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
    if use_unbinned:
        unbinned = tree["metadata/corrections"].to_dataset(inherit=False)["wavelengths_unbinned"].values
        band_centers, band_fwhm, bin_factor = unbinned, params.fwhm(unbinned), params.bin_factor
    else:
        band_centers, band_fwhm, bin_factor = wavelengths, fwhm, 1

    sr = compute_spectral_response(band_centers_unbinned=band_centers, fwhm_unbinned=band_fwhm,
                                   bin_factor=bin_factor, ssi_source=ssi_source, grid="native-truncated")
    reflectance = compute_reflectance(toa_radiance=radiance.values, sr=sr, iso_time=params.iso_time,
                                      solar_zenith_angles=root[zenith_name].values)

    out = root.drop_vars([product_name(tree), "wavelength", "radiation_wavelength", "fwhm"])
    out["rhot"] = xr.DataArray(reflectance, dims=PRODUCT_DIMS, coords={
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
