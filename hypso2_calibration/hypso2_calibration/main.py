from importlib.resources import files
import warnings

import yaml

# Which coefficient file each correction uses, per version, and the versions
# each release selects - data/coefficients.yaml (data-only edits for new
# coefficients; see that file's header).
CATALOG = yaml.safe_load(files('hypso2_calibration').joinpath('data/coefficients.yaml').read_text())

CORRECTIONS = ("radiometric", "smile", "spectral", "spectral_full_frame")


def _version_key(version: str, available) -> str:
    # "v3" and "V3" both name release/version V3.
    for key in available:
        if str(key).upper() == str(version).upper():
            return key
    raise ValueError(f"Unknown version {version!r}. Available: {sorted(available)}")


def resolve_coefficients(coeff_type: str = None, version=None) -> dict:
    """The coefficient selection for a coeff_type and version, without file
    paths: {"coeff_type", "release", "radiometric": "V3/abs", "smile": "V3",
    "spectral": "V2", "spectral_full_frame": "V2"}.

    :param coeff_type: radiometric variant ("abs", "moved", "lab");
        default CATALOG["default_coeff_type"]. The deprecated names
        "adjusted" and "original" map to (abs, V2) and (lab, V1).
    :param version: a release name ("V3", "V2", "V1"; default
        CATALOG["default_release"]), or {correction: version} to mix, e.g.
        {"radiometric": "V3", "spectral": "V3"}. Corrections missing from
        the dict come from the default release.
    """
    if coeff_type is None:
        coeff_type = CATALOG["default_coeff_type"]
    if coeff_type in CATALOG["aliases"]:
        new_type, release = CATALOG["aliases"][coeff_type]
        warnings.warn(f"coeff_type={coeff_type!r} is deprecated; use coeff_type={new_type!r}, "
                      f"version={release!r}", DeprecationWarning, stacklevel=3)
        coeff_type = new_type
        if version is None:
            version = release

    releases = CATALOG["releases"]
    if version is None or isinstance(version, str):
        release = _version_key(version or CATALOG["default_release"], releases)
        selection = dict(releases[release])
    else:
        unknown = sorted(set(version) - set(CORRECTIONS))
        if unknown:
            raise ValueError(f"Unknown corrections {unknown}. Known: {list(CORRECTIONS)}")
        release = "custom"
        selection = dict(releases[CATALOG["default_release"]])
        for correction, value in version.items():
            selection[correction] = value

    for correction in CORRECTIONS:
        kind = "spectral" if correction == "spectral_full_frame" else correction
        selection[correction] = _version_key(selection[correction], CATALOG[kind])

    variants = CATALOG["radiometric"][selection["radiometric"]]
    if coeff_type not in variants:
        raise ValueError(f"Invalid coeff_type: {coeff_type!r} for radiometric {selection['radiometric']}. "
                         f"Available: {sorted(variants)}")
    if selection["spectral"] != "V2" and release != "V1":
        warnings.warn(f"Spectral calibration {selection['spectral']} renames integer-wavelength bands "
                      f"relative to V2; OC-SMART's HYPSO band table and POLYMER band settings select bands "
                      f"by these names.", UserWarning, stacklevel=3)

    return {"coeff_type": coeff_type, "release": release,
            "radiometric": f"{selection['radiometric']}/{coeff_type}",
            "smile": selection["smile"], "spectral": selection["spectral"],
            "spectral_full_frame": selection["spectral_full_frame"]}


def get_hypso2_calibration_files(capture_type=None, coeff_type=None, version=None) -> dict:
    """
    Get the absolute paths of the calibration coefficients included in the
    package (radiometric, smile, destriping, spectral). See
    resolve_coefficients for coeff_type and version; the defaults are
    abs / V3. capture_type is accepted for interface compatibility and not
    used yet.

    :return: {"radiometric", "smile", "destriping", "spectral",
        "spectral_full_frame": path or None, "selection": resolve_coefficients(...)}.
    """
    selection = resolve_coefficients(coeff_type, version)
    radiometric_version, variant = selection["radiometric"].split("/")

    def path(name):
        return files('hypso2_calibration').joinpath(f'data/{name}')

    return {
        "radiometric": path(CATALOG["radiometric"][radiometric_version][variant]["file"]),
        "smile": path(CATALOG["smile"][selection["smile"]]["file"]),
        "destriping": None,   # no HYPSO-2 destriping coefficients in any version
        "spectral": path(CATALOG["spectral"][selection["spectral"]]["file"]),
        "spectral_full_frame": path(CATALOG["spectral"][selection["spectral_full_frame"]]["file"]),
        "selection": selection,
    }
