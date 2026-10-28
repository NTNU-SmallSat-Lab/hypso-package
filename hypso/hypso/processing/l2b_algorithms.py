"""The package's own L2B algorithms, on remote-sensing reflectance.

Each algorithm takes the L2A cube (``Rrs`` in sr-1; NDCI also ``rhos``) on
the product's bands and returns 2D products. Rrs at an algorithm's
wavelength is interpolated linearly between the two nearest bands (HYPSO's
bands are ~3.5 nm apart). Coefficients are those tabulated in ACOLITE
(``data/Shared/algorithms``), which follow the cited papers and NASA's
ATBDs:

- ``oc4``: OC4 band ratio, Sentinel-3 OLCI coefficients (O'Reilly and
  Werdell 2019): log10(chl) = sum a_i X**i,
  X = log10(max(Rrs443, Rrs490, Rrs510) / Rrs560).
- ``ci``: colour index (Hu et al. 2012; coefficients of Hu et al. 2019):
  CI = Rrs555 - (Rrs443 + (555 - 443) / (670 - 443) (Rrs670 - Rrs443)),
  chl = 10**(a0 + a1 CI).
- ``oci`` (default for CHL): NASA's chlor_a - CI below 0.25 mg m-3, OC4
  above 0.35, linear blend in between.
- ``ndci``: red-edge index (Mishra and Mishra 2012),
  NDCI = (R705 - R670) / (R705 + R670), chl = 14.039 + 86.115 NDCI +
  194.325 NDCI**2; calibrated for turbid and eutrophic waters only, no
  validity mask is applied.
- ``nechad2010`` (default for TSM): rhow = pi Rrs665,
  SPM = A rhow / (1 - rhow / C), A = 355.85 g m-3, C = 0.1728 (Nechad et
  al. 2010); NaN where rhow >= 0.5 C (ACOLITE's default limit).

Not yet: Kd(490) (needs a verified coefficient source) and IOP inversions
(QAA, TODO 4i step 4).
"""
import numpy as np

from hypso.io import cf

OC4_OLCI = (0.42540, -3.21679, 2.86907, -0.62628, -1.09333)
CI_HU2019 = (-0.4287, 230.47)
OCI_BLEND = (0.25, 0.35)
NDCI_MISHRA2012 = (14.039, 86.115, 194.325)
NECHAD2010_665 = {"wavelength": 665.0, "A": 355.85, "C": 0.1728, "max_rhow_C_factor": 0.5}

_CHL = {"units": "mg m-3", "standard_name": cf.CHLA_STANDARD_NAME}


def band_at(cube: np.ndarray, wavelengths, target: float, max_gap: float = 10.0) -> np.ndarray:
    """The cube at ``target`` nm, linearly interpolated between the nearest
    bands below and above it.

    Raises
    ------
    ValueError
        If ``target`` is outside the bands, or no band is within ``max_gap`` nm.
    """
    wavelengths = np.asarray(wavelengths, float)
    order = np.argsort(wavelengths)
    wl = wavelengths[order]
    below = np.searchsorted(wl, target, side="right") - 1
    above = min(below + 1, len(wl) - 1)
    if below < 0 or target > wl[-1]:
        raise ValueError(f"{target} nm is outside the product's bands ({wl[0]:.0f}-{wl[-1]:.0f} nm)")
    lo, hi = wl[below], wl[above]
    if min(abs(target - lo), abs(hi - target)) > max_gap:
        raise ValueError(f"no band within {max_gap} nm of {target} nm")
    weight = (target - lo) / (hi - lo) if hi > lo else 0.0
    return ((1 - weight) * cube[..., order[below]] + weight * cube[..., order[above]]).astype(np.float64)


def chl_oc4(rrs, wavelengths):
    blue = np.fmax(np.fmax(band_at(rrs, wavelengths, 443), band_at(rrs, wavelengths, 490)),
                   band_at(rrs, wavelengths, 510))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(blue / band_at(rrs, wavelengths, 560))
    return np.power(10.0, sum(a * x ** i for i, a in enumerate(OC4_OLCI)))


def chl_ci(rrs, wavelengths):
    r443, r555, r670 = (band_at(rrs, wavelengths, w) for w in (443, 555, 670))
    ci = r555 - (r443 + (555.0 - 443.0) / (670.0 - 443.0) * (r670 - r443))
    return np.power(10.0, CI_HU2019[0] + CI_HU2019[1] * ci)


def _oci(rrs, wavelengths):
    oc4, ci = chl_oc4(rrs, wavelengths), chl_ci(rrs, wavelengths)
    t1, t2 = OCI_BLEND
    weight = np.clip((ci - t1) / (t2 - t1), 0.0, 1.0)      # 0 at or below t1, 1 above t2
    chl = weight * oc4 + (1 - weight) * ci
    return {"chlor_a": (chl, {**_CHL, "long_name": "Chlorophyll-a concentration (OCI: CI and OC4 blend)"}),
            "chl_oc4": (oc4, {**_CHL, "long_name": "Chlorophyll-a concentration (OC4, OLCI coefficients)"}),
            "chl_ci": (ci, {**_CHL, "long_name": "Chlorophyll-a concentration (colour index, Hu et al.)"})}


def _oc4(rrs, wavelengths):
    return {"chl_oc4": _oci(rrs, wavelengths)["chl_oc4"]}


def _ci(rrs, wavelengths):
    return {"chl_ci": _oci(rrs, wavelengths)["chl_ci"]}


def _ndci(cube, wavelengths):
    r670, r705 = band_at(cube, wavelengths, 670), band_at(cube, wavelengths, 705)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndci = (r705 - r670) / (r705 + r670)
    a0, a1, a2 = NDCI_MISHRA2012
    return {"chl_ndci": (a0 + a1 * ndci + a2 * ndci ** 2,
                         {**_CHL, "long_name": "Chlorophyll-a concentration (NDCI, Mishra and Mishra 2012; "
                                               "turbid and eutrophic waters only)"}),
            "ndci": (ndci, {"units": "1", "long_name": "Normalized Difference Chlorophyll Index (705, 670 nm)"})}


def _nechad2010(rrs, wavelengths):
    c = NECHAD2010_665
    rhow = np.pi * band_at(rrs, wavelengths, c["wavelength"])
    with np.errstate(divide="ignore", invalid="ignore"):
        spm = c["A"] * rhow / (1.0 - rhow / c["C"])
    spm = np.where(rhow >= c["max_rhow_C_factor"] * c["C"], np.nan, spm)
    return {"spm_nechad2010_665": (spm, {"units": "g m-3", "standard_name": cf.SPM_STANDARD_NAME,
                                         "long_name": "Suspended particulate matter (Nechad et al. 2010, 665 nm)"})}


#: (group, algorithm) -> (function, cube variables it accepts, reference, default for the group)
ALGORITHMS = {
    ("CHL", "oci"): (_oci, ("Rrs",), "NASA OCI: Hu et al. 2012/2019 CI and O'Reilly and Werdell 2019 OC4", True),
    ("CHL", "oc4"): (_oc4, ("Rrs",), "O'Reilly and Werdell 2019 (OLCI OC4)", False),
    ("CHL", "ci"): (_ci, ("Rrs",), "Hu et al. 2012, coefficients Hu et al. 2019", False),
    ("CHL", "ndci"): (_ndci, ("Rrs", "rhos"), "Mishra and Mishra 2012", False),
    ("TSM", "nechad2010"): (_nechad2010, ("Rrs",), "Nechad et al. 2010", True),
}


def default_algorithm(group: str) -> str:
    """The default algorithm of an L2B group."""
    for (g, name), (*_, default) in ALGORITHMS.items():
        if g == group and default:
            return name
    raise ValueError(f"no package algorithm for group {group!r}; available: {sorted({g for g, _ in ALGORITHMS})}")
