"""Band subsets: keep some of a product's bands and drop the rest
(Product.select_bands).

From L1B on only: calibration needs every band (the smile correction resamples
each spectrum across bands), so L1A is always complete. Each kept band keeps
its index in the sensor's full band set (the ``band_index`` coordinate, and
each band variable's ``band`` attribute in files), which is what TOA
reflectance (F0) and ``metadata/srf`` (``Band_<index>``) look bands up by - so
selecting bands and then processing gives the same values as processing all
bands and selecting afterwards.
"""
import numpy as np
import xarray as xr

from hypso.io.product import BAND_INDEX, band_indices, product_name
from hypso.io.provenance import add_step

# metadata/srf per-band tables (each its own dimension); "esun" names in files
# written before 2026-10-27.
_SRF_TABLES = ("F0", "F0_wavelengths", "esun", "esun_wavelengths", "effective_fwhm")


def _positions(cube: xr.DataArray, wavelength_range, wavelengths, indices) -> np.ndarray:
    given = [x is not None for x in (wavelength_range, wavelengths, indices)]
    if sum(given) != 1:
        raise ValueError("give exactly one of wavelength_range, wavelengths or indices")
    centres = cube["radiation_wavelength"].values
    if wavelength_range is not None:
        low, high = wavelength_range
        positions = np.flatnonzero((centres >= low) & (centres <= high))
    elif wavelengths is not None:
        positions = np.array([int(np.argmin(np.abs(centres - w))) for w in np.atleast_1d(wavelengths)])
        if len(set(positions.tolist())) != positions.size:
            raise ValueError("two requested wavelengths map to the same band")
    else:
        current = band_indices(cube)
        missing = sorted(set(np.atleast_1d(indices).tolist()) - set(current.tolist()))
        if missing:
            raise ValueError(f"band indices {missing} are not in this product")
        positions = np.array([int(np.flatnonzero(current == i)[0]) for i in np.atleast_1d(indices)])
    if positions.size == 0:
        raise ValueError("the selection keeps no bands")
    return np.sort(positions)


def _subset_srf(srf: xr.Dataset, kept: np.ndarray) -> xr.Dataset:
    # The group's own bands (its Band_<i> variables, in order; the per-band
    # tables follow the same order) - which may be more than the cube's, e.g.
    # when the cube was read with a band selection.
    own = sorted(int(n.split("_")[1]) for n in srf.data_vars if n.startswith("Band_"))
    keep = set(int(i) for i in kept)
    drop = [f"{prefix}Band_{i}" for i in own if i not in keep for prefix in ("", "wav_")]
    srf = srf.drop_vars([name for name in drop if name in srf.variables])
    rows = [own.index(i) for i in sorted(keep) if i in own]
    return srf.isel({name: rows for name in _SRF_TABLES if name in srf.dims and srf.sizes[name] == len(own)})


def select_bands(tree: xr.DataTree, wavelength_range=None, wavelengths=None, indices=None) -> xr.DataTree:
    """A product tree with only some of its bands.

    Parameters
    ----------
    tree : xarray.DataTree
        L1B, L1C, L1D or L2A product tree.
    wavelength_range : tuple of float, optional
        ``(min, max)`` in nm: the bands whose centre lies within it.
    wavelengths : array_like, optional
        Wavelengths in nm: the nearest band to each.
    indices : array_like of int, optional
        Band indices in the sensor's full band set (``band_index``).

    Returns
    -------
    xarray.DataTree
        Same level, with the kept bands' ``band_index``, ``metadata/srf`` (and
        ``metadata/machi``) trimmed to them, and a ``select_bands`` step.
    """
    level = tree.attrs.get("processing_level")
    if level == "L1A":
        raise ValueError("select_bands needs L1B or later: calibration needs every band (the smile correction "
                         "resamples each spectrum across bands); calibrate first")
    cube = tree.dataset[product_name(tree)]
    current = band_indices(cube)
    positions = _positions(cube, wavelength_range, wavelengths, indices)
    kept = current[positions]

    root = tree.to_dataset(inherit=False)
    if BAND_INDEX not in root.coords:
        root = root.assign_coords({BAND_INDEX: ("bands", current)})
    if positions.size != current.size:          # all kept (e.g. read with a selection): no copy
        root = root.isel(bands=positions)

    result = tree.copy()
    # Other metadata on the full band set (metadata/corrections: the calibrated
    # wavelengths and spectral coefficients) stays complete - it records what
    # calibration used - under a "bands_full" dimension, since a child group
    # must not reuse its parent's "bands" with another size.
    for node in list(result.subtree):
        if node.is_root or node.path.endswith(("/srf", "/machi")):
            continue
        ds = node.to_dataset(inherit=False)
        if "bands" in ds.dims:
            result[node.path] = xr.DataTree(ds.rename_dims({"bands": "bands_full"}))
    result.dataset = root
    if "metadata" in tree.children and "srf" in tree["metadata"].children:
        result["metadata/srf"] = xr.DataTree(_subset_srf(tree["metadata/srf"].to_dataset(inherit=False), kept))
    if "metadata" in tree.children and "machi" in tree["metadata"].children:
        machi = tree["metadata/machi"].to_dataset(inherit=False)
        if machi.sizes.get("bands") != kept.size:
            own = band_indices(machi) if BAND_INDEX in machi.coords else np.arange(machi.sizes["bands"])
            machi = machi.isel(bands=[int(np.flatnonzero(own == i)[0]) for i in kept])
        result["metadata/machi"] = xr.DataTree(machi)
    source = tree.encoding.get("source")
    result.encoding.pop("source", None)
    selection = ({"wavelength_range": list(wavelength_range)} if wavelength_range is not None else
                 {"wavelengths": [float(w) for w in np.atleast_1d(wavelengths)]} if wavelengths is not None else
                 {"indices": [int(i) for i in np.atleast_1d(indices)]})
    add_step(result, "select_bands", input_file=source or f"{level} (in memory)",
             parameters={**selection, "bands_kept": [int(i) for i in kept], "band_count": int(kept.size)})
    return result
