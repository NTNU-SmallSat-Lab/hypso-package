"""Read a HYPSO capture's L2 files as one xarray Dataset.

A processed capture directory holds, per atmospheric correction method:

    <SAT>_HSI.<time>.<target>.L2A.RRS.<method>.nc   Rrs (sr-1); OC-SMART: also aod
    <SAT>_HSI.<time>.<target>.L2A.RHOS.<method>.nc  rhos (surface reflectance; MACHI, DPS, ACOLITE L2R)
    <SAT>_HSI.<time>.<target>.L2B.BGC.<method>.nc   concentrations: chlorophyll-a, suspended matter
    <SAT>_HSI.<time>.<target>.L2B.IOP.<method>.nc   IOPs: aph, adg, apg, bbp, bp (m-1), Kd
    <SAT>_HSI.<time>.<target>.L2A.AUX.<method>.nc   the AC tool's intermediates (Lt, Lr, ...; debugging)

Caution: OC-SMART's HYPSO-2 ``apg`` (its at) and ``bp`` (its bt) come from
networks that gave implausible values in our checks (Venice, Lake Erie,
2026-11); aph, adg and bbp were consistent at Venice.

Every file has the same pixels, geometry (latitude, longitude, angles) and
flags. This script opens them together: one Dataset, geometry and flags once,
and each per-band quantity (stored as one 2D variable per band, ``Rrs_441``)
stacked into a spectral variable (``Rrs`` with a ``wavelength`` dimension).

Needs numpy, xarray and netCDF4. The data are read into memory as numpy
arrays (a full OC-SMART capture without AUX: about 1.5 GB); ``lazy=True``
uses dask instead, reading only what is used (some operations then need an
axis, e.g. ``.median(dim=...)``, or ``.compute()`` first). Older files
(L2B.CHL, L2B.TSM, L2B.KD, L2B.AOT) are read too.

Python::

    from hypso_l2_reader import open_l2, flag_mask
    ds = open_l2("aeronetvenice_2025-03-04T10-38-05Z")       # the capture directory
    ds["Rrs"].sel(wavelength=560, method="nearest").plot()
    ds["chlor_a_oci"].where(~flag_mask(ds, "land", "cloud")).plot()
    ds["aph"].sel(wavelength_iop=443, method="nearest")
    spectrum = ds["Rrs"].isel(lines=300, samples=500)

Command line::

    python hypso_l2_reader.py CAPTURE_DIR                    # list the contents
    python hypso_l2_reader.py CAPTURE_DIR --merge out.nc     # write everything to one file

Spectral dimensions: ``wavelength`` for the reflectance (Rrs or rhos, with a
``fwhm`` coordinate), ``wavelength_iop`` for the IOPs (aph, adg, ... share
their bands), ``wavelength_aod`` for the aerosol optical depth,
``wavelength_aux`` for the intermediates. Variables with different
spectral dimensions do not broadcast against each other by accident; select
a wavelength (``.sel(..., method="nearest")``) to combine them.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

#: File kinds, in the order they are merged, and what they hold.
KINDS = {
    "RRS": "remote-sensing reflectance (and the AC tool's aerosol optical depth)",
    "RHOS": "surface reflectance",
    "RHOW": "water-leaving reflectance (older Polymer files)",
    "BGC": "concentrations: chlorophyll-a, suspended matter",
    "IOP": "inherent optical properties and Kd",
    "AUX": "the AC tool's intermediates",
    # files written before 2026-11-06, one product per file
    "CHL": "chlorophyll-a (older files)",
    "TSM": "suspended matter (older files)",
    "KD": "diffuse attenuation (older files)",
    "AOT": "aerosol optical depth (older files)",
}
REFLECTANCE = ("RRS", "RHOS", "RHOW")
DEFAULT_KINDS = ("RRS", "RHOS", "RHOW", "BGC", "IOP", "CHL", "TSM", "KD", "AOT")
_NAME = re.compile(r"^(?P<platform>[A-Z0-9]+_[A-Z0-9]+)\.(?P<time>\d{8}T\d{6})\.(?P<target>[A-Za-z0-9_-]+)\."
                   r"(?P<level>L2[AB])\.(?P<kind>[A-Z]+)\.(?P<method>[A-Za-z0-9_-]+)(?P<rest>(\.[A-Za-z0-9_-]+)*)\.nc$")
_BAND = re.compile(r"^(?P<quantity>.+)_(?P<nm>\d+)$")


def _band_free(long_name):
    """A per-band long_name without its band: "Remote Sensing Reflectance Band
    18 (441 nm)" -> "Remote Sensing Reflectance", "Absorption by phytoplankton
    (441 nm, OC-SMART)" -> "Absorption by phytoplankton (OC-SMART)"."""
    text = re.sub(r"\s*Band \d+", "", long_name)
    text = re.sub(r"\d+(\.\d+)? nm,?\s*", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    return re.sub(r"\s+", " ", re.sub(r"\(\s+", "(", text)).strip()


def find_l2_files(directory, method=None, extra=None):
    """``{kind: path}`` of one method's L2 files in a capture directory.

    Parameters
    ----------
    directory : str or path
    method : str, optional
        The atmospheric correction method (file-name token, e.g. ``"ocsmart"``,
        ``"polymer"``, ``"acolite"``, ``"machi"``). Needed only when the
        directory holds more than one.
    extra : str, optional
        The name's optional trailing fields, e.g. a grid ``"utm33n_880m"`` or
        a label; default: files without any.
    """
    found = defaultdict(dict)
    for path in sorted(Path(directory).glob("*.L2[AB].*.nc")):
        m = _NAME.match(path.name)
        if m and m["kind"] in KINDS and m["rest"].lstrip(".") == (extra or ""):
            found[m["method"]][m["kind"]] = path
    methods = {name for name, files in found.items() if set(files) & set(REFLECTANCE)}
    if method is None:
        if len(methods) != 1:
            raise ValueError(f"{directory}: {'no L2A files' if not methods else 'several methods'} "
                             f"({sorted(methods)}); pass method=")
        method = methods.pop()
    if method not in found:
        raise FileNotFoundError(f"{directory}: no L2 files for method {method!r}; found {sorted(found)}")
    return found[method]


def _wavelength(var):
    attrs = var.attrs
    return float(attrs.get("radiation_wavelength", attrs.get("wavelength")))


def _spectral_dim(kind, quantity, n_sets):
    if kind in REFLECTANCE and quantity in ("Rrs", "rhos", "rho_w"):
        return "wavelength"
    if kind in ("IOP", "AUX") and n_sets == 1:
        return f"wavelength_{kind.lower()}"
    return f"wavelength_{quantity.lower()}"


def stack_bands(ds, kind):
    """``ds`` with each quantity stored per band (``aph_441``, ``aph_445``,
    ...; at least two bands with a ``wavelength`` attribute) stacked into one
    variable (``aph``) along a spectral dimension."""
    groups = defaultdict(list)
    for name, var in ds.data_vars.items():
        m = _BAND.match(name)
        if m and ("wavelength" in var.attrs or "radiation_wavelength" in var.attrs):
            groups[m["quantity"]].append(name)
    groups = {q: sorted(names, key=lambda n: _wavelength(ds[n])) for q, names in groups.items() if len(names) > 1}
    if not groups:
        return ds
    sets = {tuple(_wavelength(ds[n]) for n in names) for q, names in groups.items()
            if not (kind in REFLECTANCE and q in ("Rrs", "rhos", "rho_w"))}
    out = ds.drop_vars([n for names in groups.values() for n in names])
    for quantity, names in groups.items():
        dim = _spectral_dim(kind, quantity, len(sets))
        wavelengths = np.array([_wavelength(ds[n]) for n in names])
        stacked = xr.Variable.concat([ds[n].variable for n in names], dim=dim)
        stacked = stacked.transpose(*ds[names[0]].dims, dim)
        first = ds[names[0]].attrs
        attrs = {k: first[k] for k in ("units", "standard_name") if k in first}
        if "long_name" in first:
            attrs["long_name"] = _band_free(first["long_name"])
        if dim in out.coords:
            if not np.allclose(out[dim].values, wavelengths):
                raise ValueError(f"{quantity}: bands differ from the others on {dim}")
        else:
            out = out.assign_coords({dim: (dim, wavelengths, {"units": "nm", "long_name": "Band centre wavelength"})})
            if dim == "wavelength" and all("fwhm" in ds[n].attrs for n in names):
                out = out.assign_coords(fwhm=(dim, np.array([float(ds[n].attrs["fwhm"]) for n in names]),
                                              {"units": "nm", "long_name": "Band full width at half maximum"}))
        stacked.attrs = attrs
        out[quantity] = stacked
    return out


def open_l2(directory, method=None, kinds=DEFAULT_KINDS, aux=False, spectra=True, extra=None, lazy=False):
    """One method's L2 files of a capture as one Dataset.

    Parameters
    ----------
    directory : str or path
        The capture directory (``<target>_<time>``) holding the files.
    method : str, optional
        AC method, e.g. ``"ocsmart"``; needed only if there are several.
    kinds : sequence of str
        Which files (:data:`KINDS`); default all but AUX. Missing ones are skipped.
    aux : bool
        Also the AUX file (the AC tool's intermediates: large, for debugging).
    spectra : bool
        Stack per-band variables into spectra (default); False keeps ``Rrs_441`` etc.
    extra : str, optional
        A regridded product's grid (``"utm33n_880m"``) or a label.
    lazy : bool
        Read with dask, only what is used (needs dask); default: into memory.

    Returns
    -------
    xarray.Dataset
        Dimensions ``lines`` and ``samples`` (or ``y``/``x`` on a grid) and
        the spectral ones; ``attrs["source_files"]`` lists the files read.
    """
    files = find_l2_files(directory, method, extra)
    wanted = list(kinds) + (["AUX"] if aux else [])
    parts, used = [], []
    for kind in KINDS:
        if kind in wanted and kind in files:
            # cache=False: stacking reads each band once, without also keeping it in ``ds``
            ds = xr.open_dataset(files[kind], chunks={} if lazy else None, cache=False)
            part = stack_bands(ds, kind) if spectra else ds
            if not lazy:
                part = part.load()
                ds.close()
            parts.append(part)
            used.append(files[kind].name)
    if not parts:
        raise FileNotFoundError(f"none of {wanted} in {sorted(files)}")
    # geometry, coordinates and flags are the same in every file: kept once (from the first);
    # assigned rather than xr.merge'd, which would copy every array
    merged = parts[0]
    for part in parts[1:]:
        if any(merged.sizes.get(d, n) != n for d, n in part.sizes.items()):
            raise ValueError(f"files of different shapes: {dict(merged.sizes)} and {dict(part.sizes)}")
        for name, coord in part.coords.items():
            if name not in merged.coords:
                merged = merged.assign_coords({name: coord})
        for name, var in part.data_vars.items():
            if name not in merged:
                merged[name] = var
    for name in ("l2b_group", "l2b_source", "l2a_quantity"):
        merged.attrs.pop(name, None)
    merged.attrs["source_files"] = ", ".join(used)
    return merged


def flag_mask(ds, *names, variable="pixel_flags"):
    """True where any of the named flags is set (CF ``flag_masks`` and
    ``flag_meanings`` of ``variable``); no names: any flag.

    >>> water = ds["chlor_a_oci"].where(~flag_mask(ds, "land", "cloud"))
    """
    if variable not in ds:
        raise KeyError(f"no {variable} in the Dataset (an unmasked product has no flags)")
    flags = ds[variable]
    meanings = flags.attrs["flag_meanings"].split()
    masks = np.atleast_1d(flags.attrs["flag_masks"])
    if isinstance(masks[0], str):                     # "[1 2 4]" as a string
        masks = np.array(masks[0].strip("[]").split(), dtype=int)
    unknown = set(names) - set(meanings)
    if unknown:
        raise KeyError(f"unknown flags {sorted(unknown)}; {variable} has {meanings}")
    bits = 0
    for meaning, mask in zip(meanings, masks):
        if not names or meaning in names:
            bits |= int(mask)
    return (flags.astype(np.int64) & bits) != 0


def _summary(ds):
    lines = [f"files: {ds.attrs['source_files']}", f"dimensions: {dict(ds.sizes)}"]
    for name, var in ds.data_vars.items():
        if var.ndim >= 2:
            spectral = [d for d in var.dims if d.startswith("wavelength")]
            extent = ""
            if spectral:
                wl = ds[spectral[0]].values
                extent = f"  {len(wl)} bands {wl.min():.0f}-{wl.max():.0f} nm"
            lines.append(f"  {name:22s} {var.attrs.get('units', ''):8s} {var.attrs.get('long_name', '')}{extent}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", help="capture directory with the L2 files")
    parser.add_argument("--method", help="AC method (needed if the directory has several)")
    parser.add_argument("--aux", action="store_true", help="include the AUX file (AC intermediates)")
    parser.add_argument("--merge", metavar="OUT.nc", help="write the merged Dataset to one netCDF file")
    args = parser.parse_args()
    ds = open_l2(args.directory, args.method, aux=args.aux, lazy=not args.merge)
    print(_summary(ds))
    if args.merge:
        encoding = {}
        for name, var in ds.data_vars.items():
            encoding[name] = {"zlib": True, "complevel": 4}
            if var.ndim == 3:                  # whole spectra of 64 x 128 pixel blocks
                encoding[name]["chunksizes"] = (min(64, var.shape[0]), min(128, var.shape[1]), var.shape[2])
        ds.to_netcdf(args.merge, encoding=encoding)
        print(f"wrote {args.merge}")


if __name__ == "__main__":
    main()
